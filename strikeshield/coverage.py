"""Threat matrix coverage metrics, grid generation, and reporting."""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from pydantic import BaseModel

from strikeshield.config import settings
from strikeshield.db import get_connection
from strikeshield.embeddings import mean_pairwise_distance
from strikeshield.taxonomy import CellId, all_cells, cell_id


class CoverageReport(BaseModel):
    cells_total: int = 40
    cells_filled: int
    coverage_ratio: float
    per_cell: Dict[CellId, int]
    mean_pairwise_distance: float
    weakest_cells: List[CellId]

    def format_table(self) -> str:
        """Renders the 5x8 threat matrix as a clean markdown table."""
        risk_ids = [r.id for r in settings.taxonomy.risk_categories]
        tech_ids = [t.id for t in settings.taxonomy.technique_families]

        header = "| Risk Category \\ Technique | " + " | ".join(tech_ids) + " | Total |"
        separator = "| :--- | " + " | ".join([":---:"] * len(tech_ids)) + " | :---: |"

        rows = [header, separator]
        for r in risk_ids:
            row_counts = [self.per_cell.get(cell_id(r, t), 0) for t in tech_ids]
            row_total = sum(row_counts)
            row_str = f"| **{r}** | " + " | ".join(str(c) for c in row_counts) + f" | **{row_total}** |"
            rows.append(row_str)

        # Technique totals footer
        tech_totals = [sum(self.per_cell.get(cell_id(r, t), 0) for r in risk_ids) for t in tech_ids]
        overall_total = sum(tech_totals)
        footer = "| **Total** | " + " | ".join(f"**{c}**" for c in tech_totals) + f" | **{overall_total}** |"
        rows.append(footer)

        summary = (
            f"\n\n**Grid Coverage Summary:**\n"
            f"- Total Attacks: {overall_total}\n"
            f"- Cells Filled: {self.cells_filled} / {self.cells_total} ({self.coverage_ratio:.1%})\n"
            f"- Mean Pairwise Cosine Distance: {self.mean_pairwise_distance:.4f}\n"
            f"- Weak Cells (<= 1 attack): {len(self.weakest_cells)}\n"
        )
        return "\n".join(rows) + summary


def get_coverage_report(db_path: Optional[Path] = None) -> CoverageReport:
    """Queries SQLite database to build full 5x8 coverage metrics."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute("SELECT cell_id, COUNT(*) as cnt FROM attacks GROUP BY cell_id;")
        per_cell_db = {row["cell_id"]: int(row["cnt"]) for row in cursor.fetchall()}

        cells = all_cells()
        per_cell = {cid: per_cell_db.get(cid, 0) for cid in cells}

        cells_filled = sum(1 for cnt in per_cell.values() if cnt > 0)
        coverage_ratio = cells_filled / len(cells) if cells else 0.0

        weakest_cells = [cid for cid, cnt in per_cell.items() if cnt <= 1]

        cursor = conn.execute("SELECT embedding FROM attacks WHERE embedding IS NOT NULL;")
        all_blobs = [row["embedding"] for row in cursor.fetchall()]
        if len(all_blobs) > 1:
            all_vecs = np.vstack([np.frombuffer(b, dtype=np.float32) for b in all_blobs])
            mean_dist = mean_pairwise_distance(all_vecs)
        else:
            mean_dist = 0.0

        return CoverageReport(
            cells_total=len(cells),
            cells_filled=cells_filled,
            coverage_ratio=coverage_ratio,
            per_cell=per_cell,
            mean_pairwise_distance=mean_dist,
            weakest_cells=weakest_cells,
        )
    finally:
        conn.close()
