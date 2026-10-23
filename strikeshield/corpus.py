"""Corpus ingestion, intra-cell semantic deduplication, stratified splitting, and DB persistence."""

import hashlib
import random
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel

from strikeshield.attack_source import AttackRecord
from strikeshield.config import settings
from strikeshield.db import get_connection
from strikeshield.embeddings import cosine_distance, embed
from strikeshield.errors import HeldoutLeakError, StrikeShieldError


class CorpusBuildResult(BaseModel):
    n_imported: int
    n_deduped: int
    n_added: int
    n_patch: int
    n_heldout: int
    cells_filled: int
    mean_pairwise_dist: float


def compute_attack_id(cell: str, prompt: str) -> str:
    """Generates deterministic attack ID based on cell and normalized prompt."""
    digest = hashlib.sha256(f"{cell}::{prompt.strip()}".encode("utf-8")).hexdigest()[:12]
    return f"atk_{digest}"


def assert_heldout_unseen_by_patcher(attacks: List[Dict[str, Any]]) -> None:
    """Hard invariant: raises HeldoutLeakError if any attack with split='heldout' is passed."""
    leaked = [atk.get("id", "unknown") for atk in attacks if atk.get("split") == "heldout"]
    if leaked:
        raise HeldoutLeakError(
            f"Invariant violation: {len(leaked)} heldout attack(s) leaked to patcher input",
            context={"leaked_ids": leaked},
        )


def _load_existing_attacks_by_cell(conn: sqlite3.Connection) -> Dict[str, List[Tuple[str, np.ndarray, str]]]:
    """Loads existing attacks grouped by cell: cell_id -> [(id, embedding, split)]."""
    cursor = conn.execute("SELECT id, cell_id, embedding, split FROM attacks;")
    result: Dict[str, List[Tuple[str, np.ndarray, str]]] = {}
    for row in cursor.fetchall():
        cid = str(row["cell_id"])
        emb_blob = row["embedding"]
        emb = (
            np.frombuffer(emb_blob, dtype=np.float32)
            if emb_blob
            else np.zeros(384, dtype=np.float32)
        )
        result.setdefault(cid, []).append((str(row["id"]), emb, str(row["split"])))
    return result


def build_corpus(
    records: List[AttackRecord],
    db_path: Optional[Path] = None,
    dedup_threshold: Optional[float] = None,
    patch_split_ratio: Optional[float] = None,
    seed: Optional[int] = None,
) -> CorpusBuildResult:
    """Ingests attack records, performs intra-cell deduplication, stratifies splits, and saves to DB."""
    threshold = dedup_threshold if dedup_threshold is not None else settings.taxonomy.grid.dedup_threshold
    split_ratio = patch_split_ratio if patch_split_ratio is not None else settings.taxonomy.grid.patch_split_ratio
    rng_seed = seed if seed is not None else settings.default_seed

    conn = get_connection(db_path)

    try:
        existing_by_cell = _load_existing_attacks_by_cell(conn)

        # Batch embed all incoming records
        prompts = [r.prompt for r in records]
        embeddings = embed(prompts) if prompts else np.empty((0, 384), dtype=np.float32)

        candidates_by_cell: Dict[str, List[Tuple[AttackRecord, np.ndarray]]] = {}
        n_deduped = 0

        # 1. Deduplication within cell
        for rec, emb in zip(records, embeddings):
            cid = rec.cell
            is_dup = False

            # Compare against existing attacks in the same cell
            if cid in existing_by_cell:
                for _, exist_emb, _ in existing_by_cell[cid]:
                    dist = cosine_distance(emb, exist_emb)
                    if dist < threshold:
                        is_dup = True
                        break

            # Compare against newly accepted candidates in this batch for the same cell
            if not is_dup and cid in candidates_by_cell:
                for _, cand_emb in candidates_by_cell[cid]:
                    dist = cosine_distance(emb, cand_emb)
                    if dist < threshold:
                        is_dup = True
                        break

            if is_dup:
                n_deduped += 1
            else:
                candidates_by_cell.setdefault(cid, []).append((rec, emb))

        # 2. Stratified split assignment per cell
        to_insert: List[Tuple[str, str, str, str, str, bytes, str, str, str]] = []
        n_added = 0

        rng = random.Random(rng_seed)

        # Sort cell IDs in stable order per taxonomy
        from strikeshield.taxonomy import all_cells
        ordered_cells = [cid for cid in all_cells() if cid in candidates_by_cell]
        # Include any remaining cells
        for cid in sorted(candidates_by_cell.keys()):
            if cid not in ordered_cells:
                ordered_cells.append(cid)

        for cid in ordered_cells:
            cell_items = candidates_by_cell[cid]
            existing_items = existing_by_cell.get(cid, [])
            existing_ids = {item[0] for item in existing_items}

            for rec, emb in cell_items:
                atk_id = rec.id or compute_attack_id(rec.cell, rec.prompt)

                if atk_id in existing_ids:
                    continue

                # Stratified assignment: 80% patch, 20% heldout using deterministic rng
                split = "patch" if rng.random() < split_ratio else "heldout"

                to_insert.append((
                    atk_id,
                    rec.prompt,
                    rec.risk_category,
                    rec.technique_family,
                    rec.target_behaviour,
                    emb.tobytes(),
                    rec.cell,
                    split,
                    rec.source,
                ))
                existing_by_cell.setdefault(cid, []).append((atk_id, emb, split))
                n_added += 1

        # 3. Insert into attacks table
        if to_insert:
            conn.executemany(
                """
                INSERT OR IGNORE INTO attacks (
                    id, prompt, risk_category, technique_family, target_behaviour,
                    embedding, cell_id, split, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                to_insert,
            )
            conn.commit()

        # 4. Compute aggregate stats from attacks table
        cursor = conn.execute("SELECT split, COUNT(*) as cnt FROM attacks GROUP BY split;")
        split_counts = {row["split"]: row["cnt"] for row in cursor.fetchall()}
        n_patch = split_counts.get("patch", 0)
        n_heldout = split_counts.get("heldout", 0)

        cursor = conn.execute("SELECT COUNT(DISTINCT cell_id) as filled FROM attacks;")
        cells_filled = cursor.fetchone()["filled"]

        # Mean pairwise distance across all stored embeddings
        cursor = conn.execute("SELECT embedding FROM attacks WHERE embedding IS NOT NULL;")
        all_blobs = [row["embedding"] for row in cursor.fetchall()]
        if len(all_blobs) > 1:
            all_vecs = np.vstack([np.frombuffer(b, dtype=np.float32) for b in all_blobs])
            from strikeshield.embeddings import mean_pairwise_distance
            mean_dist = mean_pairwise_distance(all_vecs)
        else:
            mean_dist = 0.0

        return CorpusBuildResult(
            n_imported=len(records),
            n_deduped=n_deduped,
            n_added=n_added,
            n_patch=n_patch,
            n_heldout=n_heldout,
            cells_filled=cells_filled,
            mean_pairwise_dist=mean_dist,
        )

    finally:
        conn.close()
