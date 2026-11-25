# How much do critical points depend on grid resolution?

Date: 2026-11-25. **Tested** with `python3 scripts/eval_topology.py --fields 40 --seed 1`.

Forty random smooth wind fields (Fourier modes, length scale about 1.5 degrees) over a 10 x 10 degree box. The points
found on a 0.05 degree grid are the reference; each coarser grid is scored against them, ignoring points within one
coarse cell of the edge. A match needs the same family (node, saddle, focus) within one coarse cell.

| Grid | Reference points | Found | Recall | Precision | Exact-kind accuracy among matches |
|---|---|---|---|---|---|
| 0.25 deg | 234 | 235 | 0.97 | 0.97 | 0.99 |
| 0.5 deg | 211 | 206 | 0.93 | 0.95 | 0.98 |
| 1.0 deg | 154 | 151 | 0.83 | 0.85 | 0.94 |
| 2.5 deg | 70 | 50 | 0.47 | 0.66 | 0.91 |

## What it means

- At 0.5 degrees (the regional default) the detector recovers about nine in ten points, and classifies almost all
  of them correctly.
- At 2.5 degrees it misses half, and a third of what it finds has no counterpart. The global 5 degree default is
  coarser still: treat its critical points as the planet's largest circulation features only.
- When a point is found, its kind is reliable even on coarse grids (0.91 or better). The losses are missed and
  merged points, not misclassified ones.

## Limits

These fields are smoother than real 10 m wind, which has fronts, sea breezes and terrain channelling below 1 degree.
Real-world recall at each resolution will be lower. The next step is the same comparison on ERA5 itself: CDS at 0.25
degrees as the reference against the same hours sampled at 0.5, 1 and 2.5 degrees.
