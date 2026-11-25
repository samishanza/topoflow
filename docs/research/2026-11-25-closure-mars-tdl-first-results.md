# First results: Poincaré-Hopf closure, resolution, Mars, and topological deep learning

Date: 2026-11-25. **Tested** on real data; numbers are copied from the runs named below. Everything here is a first
look on a partial record (ERA5 months still streaming), not a final result.

## 1. Closure holds as an identity, and departs from 2 only where the grid cannot resolve the flow

For every 6-hourly step on every grid tried, the global index sum satisfies

    sum of cell winding indices + polar caps  ==  2 + edge_error

where `edge_error` counts grid edges whose turning is read in opposite senses by the two cells that share them (the
wind turns by about half a turn between neighbouring grid points). On a fine enough grid there are no such edges and
the sum is exactly chi(S^2) = 2. Located, Jacobian-classified points carry exactly their cells' winding, so the
point-based sum and the winding sum agree on every step.

**Resolution dependence** (`scripts/sensitivity_resolution.py`, 48 random 2019 times, ERA5 0.25 deg via ARCO-ERA5,
subsampled):

| Grid | Steps with sum exactly 2 | Mean sum | Mean points per step |
|---|---|---|---|
| 0.25 deg | 100% | 2.000 | 3,049 |
| 0.5 deg | 95.8% | 1.958 | 2,388 |
| 1 deg | 100% (48 steps) | 2.000 | 1,393 |
| 2 deg | 81.3% | 1.813 | 697 |
| 5 deg (Open-Meteo ERA5 sample, July 2025, 28 steps) | 10.7% | -0.500 | 265 |

On larger samples at 1 deg (2019-01 to 2019-06, 724 steps) the share exactly 2 is 82-90% per month. The number of
points per step depends strongly on resolution (4x between 0.25 and 2 deg), so counts must never be compared across
grids. Departures are always negative on average: coarse grids lose saddles more often than nodes and foci.

## 2. Persistence concentrates over terrain, on both planets

Earth (1 deg, 2019): the most persistent stagnation cells are in enclosed basins and high terrain: Tian Shan /
Dzungarian basin (44N 81E), Pamirs (37N 73E), the Andes. Near-surface winds there are weak and variable.

Mars (OpenMARS MY34, 5 deg, lowest sigma level): the most persistent cells sit at 17.5N 135W and 22.5N 130W, next to
Olympus Mons (18.7N 133.8W), and across Tharsis.

Implication for CH4: any CH4 enhancement in stagnation zones must be tested against terrain, or it will be a terrain
effect. The CH4 analysis bands by latitude and separates land and ocean; orography is a covariate in the model.

## 3. Mars closes more often than Earth at the same coarse resolution

At 5 deg, Mars Year 34 (first 3 files, 1,080 steps) has the sum exactly 2 in 78.8% of steps (mean 1.756) with about
104 points per step; Earth at 5 deg had 10.7% (28 steps). Mars's near-surface flow is much smoother at large scale.
**Caveat:** the Earth 5 deg sample is small, from a different route (Open-Meteo point sampling), and a different
season. The fair comparison is Earth ERA5 subsampled to 5 deg over a full year; that is queued.

## 4. Topological deep learning: first smoke test

Task: will a stagnation point that has persisted at least 24 h lie within 200 km of each icosphere vertex 24 h from
now? Mesh: icosphere level 5 (10,242 vertices, 30,720 edges; V - E + F = 2, B1 B2 = 0, Hodge L1 has no kernel).
Wind enters as a 1-cochain (flow along edges). One epoch, train 2019-01..03, validate 2019-04, test 2019-05..06
(236 steps x 10,242 vertices, base rate 5.1%):

| Model | Average precision | ROC-AUC | Brier |
|---|---|---|---|
| persistence | 0.126 | 0.632 | 0.0709 |
| climatology | 0.272 | 0.805 | 0.0429 |
| logistic regression | 0.291 | 0.831 | 0.0414 |
| vertex graph network (no simplicial branch) | 0.353 | 0.880 | 0.0392 |
| TopoNet (simplicial Hodge convolutions on the wind cochain) | 0.351 | 0.878 | 0.0393 |

Both networks beat every baseline clearly. **The simplicial edge branch does not yet add skill over the vertex-only
ablation.** This is one epoch on three months; the full run uses 2019-2024. If the ablation still matches TopoNet
then, the honest conclusion is that the topology of the wind cochain is not adding information beyond local wind
speed and the current stagnation pattern for this target.

## Update: the full 2010-2024 record at 1 degree

**Tested** (ERA5 via ARCO-ERA5, every 6 hours, 180 months, 21,916 steps; computed from each month's
`closure.parquet`). The index sum is exactly 2 in **88.0%** of steps (mean 1.872), and `sum = 2 + edge_error` holds on
**every** step. The share is stable year to year (86.4% in 2012 to 89.2% in 2017), with a mean of 1,369 stagnation
points per step (1,348 to 1,396 by year). On this record, departures from closure at 1 degree are a steady
resolution effect, not a drifting one.

## Not yet done

- CH4 association (composites, block bootstrap, FDR, 0-3 day lags): waiting on access to CAMS EAC4 (ADS terms must be
  accepted by the account owner) or TROPOMI (Earth Engine project).
- Full 2010-2024 record (streaming, about 2 minutes per month).
- Earth vs Mars at matched resolution.
