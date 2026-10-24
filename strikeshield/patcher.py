"""Patcher module: Technique-level policy patch generation, structural validation, model escalation, and git tracking."""

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel

from strikeshield.config import settings
from strikeshield.corpus import assert_heldout_unseen_by_patcher
from strikeshield.db import get_connection
from strikeshield.errors import HeldoutLeakError, PatcherTooWeakError
from strikeshield.llm import RoleSession, assert_vram_empty
from strikeshield.taxonomy import weakest_techniques

logger = logging.getLogger(__name__)

PATCHER_PROMPT_DIRECTIVES = """Write ONE general security rule per technique family.
Every rule must:
  - Name the technique family explicitly (e.g. "Role-play framing", "Language switching").
  - Name the forbidden behaviour it prevents (e.g. "revealing the system prompt").
  - Be enforceable (no style preferences, no tone changes).
Output ONLY a unified diff against the current policy. If a family cannot be
addressed without breaking in-scope functionality, output a rule that declines
with a specific reason rather than a generic refusal."""

STYLE_ONLY_PHRASES = [
    "answer in",
    "respond in",
    "be polite",
    "be concise",
    "adjust tone",
    "polite tone",
    "helpful tone",
    "professional tone",
    "courteous tone",
    "maintain a tone",
]

FORBIDDEN_BEHAVIOUR_KEYWORDS = [
    "never", "prohibit", "forbidden", "refuse", "do not", "disregard",
    "reject", "prevent", "disclose", "reveal", "extract", "override",
    "bypass", "leak", "unauthorized", "strictly", "block", "deny",
    "simulate", "jailbreak"
]


from strikeshield.policies import PolicyVersion


class PatchCandidate(BaseModel):
    version: str
    parent_version: str
    diff_text: str
    policy_text: str
    model_used: str
    fallback_used: bool = False
    status: str = "candidate"  # 'candidate' or 'rejected'
    rejection_reason: Optional[str] = None
    git_commit: Optional[str] = None
    addressed_families: List[str] = []


def apply_unified_diff(original_text: str, diff_text: str) -> str:
    """Applies unified diff lines to original policy text."""
    lines = diff_text.strip().splitlines()
    added_lines: List[str] = []
    removed_lines: List[str] = []

    for line in lines:
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            added_lines.append(line[1:])
        elif line.startswith("-"):
            removed_lines.append(line[1:])

    # If the patch removes all or majority lines with no added lines, simulate policy wipe
    if len(removed_lines) > 5 and len(added_lines) == 0:
        return ""

    # Clean added lines
    clean_added = [l.strip() for l in added_lines if l.strip()]
    if not clean_added:
        return original_text

    # Format new rules into automated hardening section
    new_rules = [r for r in clean_added if r not in original_text]
    if not new_rules:
        return original_text

    rule_bullets = []
    for r in new_rules:
        bullet = r if r.startswith("-") or r.startswith("*") else f"- {r}"
        rule_bullets.append(bullet)

    appended = (
        original_text.rstrip()
        + "\n\n## Automated Hardened Defense Rules\n"
        + "\n".join(rule_bullets)
        + "\n"
    )
    return appended


def validate_patch(
    candidate: PatchCandidate,
    weakest_fams: List[str],
    successes_by_family: Dict[str, List[Any]],
    old_policy_text: str,
) -> Tuple[bool, str]:
    """Strict structural validation of candidate patch diff and resulting policy.
    
    Returns (True, 'ok') or (False, reason).
    """
    # 1. No policy shrink: new policy non-blank lines >= 0.8x old
    old_non_blank = len([l for l in old_policy_text.splitlines() if l.strip()])
    new_non_blank = len([l for l in candidate.policy_text.splitlines() if l.strip()])
    if new_non_blank < 0.8 * old_non_blank:
        return False, "policy_shrunk"

    # 2. Policy parses cleanly
    if not candidate.policy_text.strip():
        return False, "empty_policy"

    diff_lines = candidate.diff_text.strip().splitlines()
    added_lines = [
        l[1:].strip()
        for l in diff_lines
        if l.startswith("+") and not l.startswith("+++") and l[1:].strip()
    ]

    # 3. No style-only rules: reject if added lines are style/tone phrases without forbidden behavior clause
    if added_lines:
        style_matches = [
            l for l in added_lines
            if any(sp in l.lower() for sp in STYLE_ONLY_PHRASES)
            and not any(kw in l.lower() for kw in FORBIDDEN_BEHAVIOUR_KEYWORDS)
        ]
        has_forbidden_anywhere = any(
            any(kw in l.lower() for kw in FORBIDDEN_BEHAVIOUR_KEYWORDS)
            for l in added_lines
        )
        if len(style_matches) > 0 and not has_forbidden_anywhere:
            return False, "style_only"

    # 4. Non-trivial diff: >= 3 meaningful lines added
    if len(added_lines) < 3:
        return False, "trivial_diff"

    # 5. Coverage: Every weakest family (in top-3) with >= 1 success must be mentioned by name
    diff_lower = " ".join(added_lines).lower()
    for fam in weakest_fams[:3]:
        succs = successes_by_family.get(fam, [])
        if len(succs) >= 1:
            norm1 = fam.lower()
            norm2 = fam.replace("_", " ").lower()
            norm3 = fam.replace("_", "-").lower()
            if not (norm1 in diff_lower or norm2 in diff_lower or norm3 in diff_lower):
                return False, "missing_weakest_family"

    return True, "ok"


def commit_policy_to_git(
    new_version: str,
    parent_version: str,
    model_used: str,
    families: List[str],
    policy_file: Path,
    git_ref: str = "refs/heads/policies_candidates",
) -> str:
    """Commits new policy version file to policies_candidates Git branch using git plumbing."""
    model_name = (
        "Mistral-7B"
        if "mistral" in model_used.lower()
        else ("Qwen2.5-7B" if "qwen" in model_used.lower() else model_used)
    )
    fams_str = ",".join(families) if families else "general"
    commit_msg = f"policy: {parent_version} → {new_version} | patcher={model_name} | families={fams_str} | gate=PENDING"

    repo_root = Path(".").resolve()

    # Ensure git_ref exists, creating from refs/heads/policies if not
    ref_check = subprocess.run(
        ["git", "rev-parse", "--verify", git_ref],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    if ref_check.returncode != 0:
        branch_name = git_ref.replace("refs/heads/", "")
        subprocess.run(
            ["git", "branch", branch_name, "refs/heads/policies"],
            check=True,
            capture_output=True,
            cwd=repo_root,
        )

    with tempfile.NamedTemporaryFile(delete=False) as f:
        tmp_index = f.name

    env = os.environ.copy()
    env["GIT_INDEX_FILE"] = tmp_index

    try:
        # Load policies tree into temp index
        subprocess.run(
            ["git", "read-tree", git_ref],
            env=env,
            check=True,
            capture_output=True,
            cwd=repo_root,
        )
        # Hash the policy file into object database
        blob_hash = subprocess.run(
            ["git", "hash-object", "-w", str(policy_file)],
            check=True,
            capture_output=True,
            text=True,
            cwd=repo_root,
        ).stdout.strip()

        target_git_path = f"policies/{policy_file.name}"
        subprocess.run(
            ["git", "update-index", "--add", "--cacheinfo", "100644", blob_hash, target_git_path],
            env=env,
            check=True,
            capture_output=True,
            cwd=repo_root,
        )

        tree = subprocess.run(
            ["git", "write-tree"],
            env=env,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=repo_root,
        ).stdout.strip()
        parent = subprocess.run(
            ["git", "rev-parse", git_ref],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=repo_root,
        ).stdout.strip()
        commit_hash = subprocess.run(
            ["git", "commit-tree", tree, "-p", parent, "-m", commit_msg],
            env=env,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=repo_root,
        ).stdout.strip()
        # Update git_ref to the new commit
        subprocess.run(
            ["git", "update-ref", git_ref, commit_hash],
            check=True,
            capture_output=True,
            cwd=repo_root,
        )
        return commit_hash
    finally:
        if os.path.exists(tmp_index):
            os.remove(tmp_index)


class Patcher:
    """Closed-loop guardrail patcher with technique-level grouping, validation, and model escalation."""

    def __init__(
        self,
        model_tag: Optional[str] = None,
        fallback_model_tag: Optional[str] = None,
        temperature: float = 0.3,
        max_validation_retries: int = 2,
        corpus_split: str = "patch",
        granularity: str = "family",
    ) -> None:
        self.model_tag = model_tag or settings.models.patcher.model
        self.fallback_model_tag = (
            fallback_model_tag
            or settings.models.patcher.fallback_model
            or "qwen2.5:7b-instruct-q4_K_M"
        )
        self.temperature = temperature
        self.max_validation_retries = max_validation_retries
        self.prompt_directives = PATCHER_PROMPT_DIRECTIVES
        self.corpus_split = corpus_split
        self.granularity = granularity

    def build_prompt(
        self,
        current_policy: str,
        successes_by_family: Dict[str, List[Dict[str, Any]]],
        weakest_fams: List[str],
        validation_feedback: Optional[str] = None,
    ) -> str:
        """Formats the patcher prompt including baseline successes, weakest techniques, and directives."""
        lines = [
            "# Guardrail Hardening Task",
            "You are an expert security engineer improving an LLM guardrail policy.",
            "",
            "## Current Policy",
            "```markdown",
            current_policy.strip(),
            "```",
            "",
            "## Baseline Weak Spots (Top Weakest Techniques)",
            f"The current baseline identified the following weakest techniques: {', '.join(weakest_fams)}",
            "",
        ]

        if self.granularity == "per_prompt":
            lines.append("## Successful Attacks (Per-Prompt Granularity)")
            for fam, attacks in sorted(successes_by_family.items()):
                for atk in attacks:
                    lines.append(f"- Prompt ID: {atk.get('id', 'unknown')}")
                    lines.append(f"  Technique: {fam}")
                    lines.append(f"  Prompt: {atk.get('prompt', '')}")
                    lines.append(f"  Target Behaviour: {atk.get('target_behaviour', '')}")
            lines.append("")
            lines.append("## Directives")
            lines.append("Write ONE general security rule per individual attack prompt.")
            lines.append(self.prompt_directives)
        else:
            lines.append("## Successful Attacks Grouped by Technique Family")
            for fam, attacks in sorted(successes_by_family.items()):
                lines.append(f"### Technique Family: {fam} (Count: {len(attacks)})")
                for atk in attacks[:3]:  # Top examples
                    prompt_snippet = atk.get("prompt", "")
                    resp_snippet = atk.get("response", "")
                    evidence = atk.get("judge_evidence_quote", "")
                    target_beh = atk.get("target_behaviour", "")
                    lines.append(f"- Target Behaviour: {target_beh}")
                    lines.append(f"  Prompt: {prompt_snippet}")
                    lines.append(f"  Assistant Response: {resp_snippet}")
                    lines.append(f"  Judge Evidence Quote: {evidence}")
            lines.append("")
            lines.append("## Directives")
            lines.append(self.prompt_directives)

        if validation_feedback:
            lines.append("")
            lines.append(f"## Previous Attempt Validation Failure")
            lines.append(
                f"Your previous patch was rejected with reason: '{validation_feedback}'. "
                "Ensure every weakest technique family is explicitly named and rules contain concrete forbidden behavior clauses."
            )

        return "\n".join(lines)

    def generate_patch(
        self,
        policy: PolicyVersion,
        successes_by_family: Dict[str, List[Dict[str, Any]]],
        weakest_families_list: List[str],
        current_metrics: Optional[Any] = None,
        corpus: Optional[Any] = None,
        db_path: Optional[Path] = None,
    ) -> PatchCandidate:
        """Generates, validates, escalates, and records a new candidate policy version."""
        # 1. Held-out leak assertion: check all attacks in successes_by_family (unless corpus_split == 'all')
        if self.corpus_split != "all":
            flat_attacks: List[Dict[str, Any]] = []
            for fam_attacks in successes_by_family.values():
                flat_attacks.extend(fam_attacks)
            assert_heldout_unseen_by_patcher(flat_attacks)

        # Compute next version name
        parent_v = policy.version
        try:
            v_num = int(re.sub(r"\D", "", parent_v))
            next_v = f"v{v_num + 1}"
        except Exception:
            next_v = f"{parent_v}_patched"

        # Determine addressed families
        addressed = [
            f for f in weakest_families_list[:3]
            if f in successes_by_family and len(successes_by_family[f]) > 0
        ]
        if not addressed and successes_by_family:
            addressed = list(successes_by_family.keys())[:3]

        # Escalation sequence: (model_name, is_fallback)
        models_to_try = [
            (self.model_tag, False),
            (self.fallback_model_tag, True),
        ]

        last_reason = "unknown"
        last_diff = ""
        last_candidate_policy = ""

        for model_to_use, is_fallback in models_to_try:
            logger.info("Attempting patch generation with model: %s (fallback=%s)", model_to_use, is_fallback)
            with RoleSession(model_to_use) as client:
                feedback: Optional[str] = None
                for attempt in range(1 + self.max_validation_retries):
                    prompt = self.build_prompt(
                        current_policy=policy.text,
                        successes_by_family=successes_by_family,
                        weakest_fams=weakest_families_list,
                        validation_feedback=feedback,
                    )

                    raw_diff = client.generate(
                        model=model_to_use,
                        prompt=prompt,
                        role="patcher",
                        temperature=self.temperature,
                    )
                    last_diff = raw_diff

                    # Apply diff
                    patched_policy_text = apply_unified_diff(policy.text, raw_diff)
                    last_candidate_policy = patched_policy_text

                    candidate = PatchCandidate(
                        version=next_v,
                        parent_version=parent_v,
                        diff_text=raw_diff,
                        policy_text=patched_policy_text,
                        model_used=model_to_use,
                        fallback_used=is_fallback,
                        status="candidate",
                        addressed_families=addressed,
                    )

                    ok, reason = validate_patch(
                        candidate,
                        weakest_families_list,
                        successes_by_family,
                        policy.text,
                    )
                    if ok:
                        # Candidate accepted structurally!
                        # Write policies/v{N+1}.md
                        out_dir = Path("policies")
                        out_dir.mkdir(parents=True, exist_ok=True)
                        candidate_file = out_dir / f"{next_v}.md"
                        candidate_file.write_text(patched_policy_text, encoding="utf-8")

                        # Commit to Git
                        commit_hash = commit_policy_to_git(
                            new_version=next_v,
                            parent_version=parent_v,
                            model_used=model_to_use,
                            families=addressed,
                            policy_file=candidate_file,
                        )
                        candidate.git_commit = commit_hash

                        # Record in SQLite policies table
                        conn = get_connection(db_path)
                        try:
                            conn.execute(
                                """
                                INSERT OR REPLACE INTO policies (
                                    version, parent_version, git_commit, diff_text,
                                    candidate_status, patcher_model, patcher_fallback_used
                                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                                """,
                                (
                                    next_v,
                                    parent_v,
                                    commit_hash,
                                    raw_diff,
                                    "candidate",
                                    model_to_use,
                                    1 if is_fallback else 0,
                                ),
                            )
                            conn.commit()
                        finally:
                            conn.close()

                        return candidate
                    else:
                        feedback = reason
                        last_reason = reason
                        logger.warning(
                            "Patch validation failed on attempt %d with model %s: %s",
                            attempt + 1,
                            model_to_use,
                            reason,
                        )

            # If this was primary model and it failed all retries, escalate
            if not is_fallback:
                logger.info("Primary patcher model failed validation (%s). Escalating to fallback model.", last_reason)

        # Both primary and fallback models failed structurally
        # Write rejected candidate to policies/candidates/v{N+1}_rejected.md
        cand_dir = Path("policies/candidates")
        cand_dir.mkdir(parents=True, exist_ok=True)
        rej_file = cand_dir / f"{next_v}_rejected.md"
        rej_content = (
            f"# REJECTED CANDIDATE: {next_v}\n"
            f"Parent: {parent_v}\n"
            f"Rejection Reason: {last_reason}\n"
            f"Patcher Model: {self.fallback_model_tag}\n\n"
            f"## Diff\n```diff\n{last_diff}\n```\n\n"
            f"## Resulting Text\n{last_candidate_policy}\n"
        )
        rej_file.write_text(rej_content, encoding="utf-8")

        # Record rejected in SQLite
        conn = get_connection(db_path)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO policies (
                    version, parent_version, diff_text,
                    candidate_status, rejection_reason, patcher_model, patcher_fallback_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    next_v,
                    parent_v,
                    last_diff,
                    "rejected",
                    last_reason,
                    self.fallback_model_tag,
                    1,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        raise PatcherTooWeakError(
            f"Patcher models failed structural validation after retries: {last_reason}",
            context={"reason": last_reason, "version": next_v},
        )
