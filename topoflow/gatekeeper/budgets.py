"""Per-agent budgets and the disk guard. An agent that loops is a cost problem and a sign its sources aren't answering."""
from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path

from .. import grid
from ..contract import BudgetBreach, RunRequest


@dataclass(frozen=True)
class Budget:
    max_tool_calls: int
    max_http_calls: int
    max_bytes: int
    max_wall_s: float


def budgets_for(req: RunRequest) -> dict[str, Budget]:
    """Budgets scale with the grid: one Open-Meteo request per 100 cells, per gridded source."""
    cells = len(grid.points(req))
    batches = math.ceil(cells / 100)
    return {
        "context": Budget(4, 40, 30_000_000, 300),
        "wind": Budget(6, batches + 10, 400_000_000, 1200),
        "methane": Budget(8, batches + 12, 200_000_000, 1200),
        "emissions": Budget(8, 40, 120_000_000, 900),
        "derive": Budget(0, 0, 0, 300),
    }


def check_budget(agent_id: str, budget: Budget, tool_calls: int, http_calls: int, bytes_: int, wall_s: float) -> BudgetBreach | None:
    for limit, allowed, used in (
        ("max_tool_calls", budget.max_tool_calls, tool_calls),
        ("max_http_calls", budget.max_http_calls, http_calls),
        ("max_bytes", budget.max_bytes, bytes_),
        ("max_wall_s", budget.max_wall_s, wall_s),
    ):
        if used > allowed:
            return BudgetBreach(agent_id=agent_id, limit=limit, allowed=allowed, used=used)
    return None


class DiskTooFull(RuntimeError):
    pass


# This project's machine has run out of disk mid-build before. A run refuses to start below this.
MIN_FREE_MB = 400


def estimated_output_mb(req: RunRequest) -> float:
    lats, lons = grid.axes(req)
    t = len(grid.times(req))
    # About 9 time-varying float32 grids, compressed roughly 2x, plus a few MB of points and manifest.
    return 9 * t * len(lats) * len(lons) * 4 / 2 / 1e6 + 5


def check_disk(path: str | Path, req: RunRequest, min_free_mb: float = MIN_FREE_MB) -> float:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    free_mb = shutil.disk_usage(p).free / 1e6
    need = estimated_output_mb(req) + 60  # plus EDGAR's transient zip and unzip
    if free_mb - need < min_free_mb:
        raise DiskTooFull(f"{free_mb:.0f} MB free; this run needs about {need:.0f} MB and keeps {min_free_mb} MB spare. "
                          f"Free some space or pick a coarser grid / shorter window.")
    return free_mb
