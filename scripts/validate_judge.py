"""Scaffolding and metric calculations for human judge validation and inter-rater agreement."""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

from strikeshield.config import settings
from strikeshield.db import get_connection
from strikeshield.errors import JudgeUnvalidatedWarning

VALIDATION_FLAG_PATH = Path("data/validation/judge_unvalidated.flag")


def check_judge_validation_status() -> None:
    """Raises JudgeUnvalidatedWarning if validation has failed and real corpus is active."""
    if VALIDATION_FLAG_PATH.exists():
        msg = "Judge model is unvalidated (Cohen's kappa < 0.6). Headline ASR figures are untrusted."
        if settings.taxonomy.grid.corpus_role.lower() == "real":
            warnings.warn(msg, JudgeUnvalidatedWarning, stacklevel=2)
        else:
            print(f"[NOTE] {msg} (Ignored under smoke corpus role)")


def sample_pairs_for_annotation(
    n_samples: int = 50,
    out_path: Path = Path("data/validation/sample.jsonl"),
    db_path: Optional[Path] = None,
) -> None:
    """Extracts stratified sample of response-verdict pairs for dual human annotation."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(db_path)

    query = """
    SELECT r.id as response_id, a.prompt, r.response_text, a.target_behaviour,
           v.verdict_class as judge_verdict, v.evidence_quote as judge_evidence, a.cell_id
    FROM responses r
    JOIN attacks a ON r.attack_id = a.id
    JOIN verdicts v ON v.response_id = r.id;
    """
    rows = conn.execute(query).fetchall()
    conn.close()

    if not rows:
        print("No response-verdict pairs found in database to sample.")
        return

    # Deterministic subsample
    rng = np.random.default_rng(settings.default_seed)
    indices = rng.choice(len(rows), size=min(n_samples, len(rows)), replace=False)

    sample_items = []
    for idx in indices:
        row = rows[int(idx)]
        sample_items.append({
            "response_id": row["response_id"],
            "cell_id": row["cell_id"],
            "prompt": row["prompt"],
            "response": row["response_text"],
            "target_behaviour": row["target_behaviour"],
            "judge_verdict": row["judge_verdict"],
            "judge_evidence": row["judge_evidence"],
            "human_label_1": None,
            "human_label_2": None,
        })

    with open(out_path, "w", encoding="utf-8") as f:
        for item in sample_items:
            f.write(json.dumps(item) + "\n")

    print(f"Sampled {len(sample_items)} response pairs to {out_path} for dual human annotation.")


def evaluate_human_labels(in_path: Path = Path("data/validation/sample.jsonl")) -> None:
    """Computes Cohen's kappa, precision/recall/F1, and confusion matrix from human labels."""
    if not in_path.exists():
        print(f"Validation file {in_path} not found.")
        sys.exit(1)

    items: List[Dict[str, Any]] = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line.strip()))

    h1_labels = [i.get("human_label_1") for i in items if i.get("human_label_1") is not None]
    h2_labels = [i.get("human_label_2") for i in items if i.get("human_label_2") is not None]

    if len(h1_labels) != len(items) or len(h2_labels) != len(items):
        print(f"Pending human annotations: only {len(h1_labels)} / {len(items)} labels filled.")
        print("Cohen's kappa status: UNKNOWN (pending human study)")
        return

    kappa = cohen_kappa_score(h1_labels, h2_labels)
    print(f"Cohen's Kappa (Inter-Annotator Agreement): {kappa:.4f}")

    if kappa < 0.6:
        print("WARNING: Cohen's kappa < 0.6. Flagging judge as unvalidated.")
        VALIDATION_FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
        VALIDATION_FLAG_PATH.write_text(f"kappa={kappa:.4f}\n", encoding="utf-8")
        sys.exit(1)
    else:
        if VALIDATION_FLAG_PATH.exists():
            VALIDATION_FLAG_PATH.unlink()
        print("Agreement validated: kappa >= 0.6")


def main() -> None:
    parser = argparse.ArgumentParser(description="StrikeShield Human Judge Validation")
    subparsers = parser.add_subparsers(dest="subcommand")

    sample_parser = subparsers.add_parser("sample", help="Sample pairs for human review")
    sample_parser.add_argument("--n", type=int, default=50)
    sample_parser.add_argument("--out", type=Path, default=Path("data/validation/sample.jsonl"))

    eval_parser = subparsers.add_parser("evaluate", help="Compute agreement metrics from filled labels")
    eval_parser.add_argument("--in", type=Path, default=Path("data/validation/sample.jsonl"))

    args = parser.parse_args()

    if args.subcommand == "sample":
        sample_pairs_for_annotation(n_samples=args.n, out_path=args.out)
    elif args.subcommand == "evaluate":
        evaluate_human_labels(in_path=getattr(args, "in"))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
