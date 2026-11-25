# Context

Living context for TopoFlow. Update it when new information arrives: add, don't silently rewrite. Date each change and
mark each statement **Confirmed** (checked against data or a primary source), **Assumed**, or **Open**.

## What TopoFlow is for (2026-11-25)

TopoFlow began as a single Dash script that drew ERA5 10 m wind arrows over a methane field, with empty slots for
stagnation points (nodes, saddles, foci). It is now a pipeline that compiles that picture from many public sources,
fills the stagnation points with a tested detector, and keeps every value traced to where it came from.

The question it serves: **where does near-surface flow converge, and is methane higher there?** Convergent critical
points (attracting nodes and foci) and negative divergence mark where surface air piles up; under a shallow boundary
layer, emitted CH4 should accumulate there. TopoFlow puts the flow topology, the CH4 field, the boundary layer, the
emission inventories, and observed plumes on one grid and one clock, so that relationship can be looked at honestly.

- **Confirmed:** the pipeline runs end to end on real public data for the Permian Basin (0.5 degrees, 2025-07-01..07) and
  globally (5 degrees). See the run summaries in [data-flow.md](data-flow.md).
- **Open:** whether CH4 is systematically higher at attracting critical points. The first Permian week shows attracting
  nodes +48 ppb and attracting foci +32 ppb above the regional mean, but saddles +39 ppb and repelling nodes +75 ppb too,
  on few points (n = 5 to 41). That is description, not evidence. It needs longer windows, finer grids, and a test
  designed before looking.

## The science it leans on

- **Critical points of a 2-D vector field.** Zeros of (u, v), classified by the Jacobian's determinant and trace:
  saddle (det < 0), node or focus (det > 0; focus when the eigenvalues are complex), attracting when the trace
  (local divergence) is negative. Detected on the piecewise-linear interpolant, triangle by triangle. **Confirmed** on
  analytic fields in `tests/`.
- **Resolution matters.** Against a 0.05 degree reference, recall is 0.93 at 0.5 degrees and 0.47 at 2.5 degrees
  ([eval](../research/2026-11-25-topology-resolution-eval.md)). **Confirmed** on synthetic smooth fields; real wind is
  rougher, so these are upper bounds.
- **Mass to mole fraction.** CAMS on Open-Meteo serves CH4 in ug m-3. TopoFlow converts with ERA5 surface pressure and
  2 m temperature per cell and time, ignoring water vapour (a few percent low in humid air). **Confirmed** formula;
  the Permian regional mean comes out 1.22x the NOAA marine background, which is plausible for an oil and gas basin
  but not independently verified.

## Sources

The registry (`topoflow/sources/registry.py`) is the list of record. Test status and terms:
[2026-11-25-ch4-wind-datasets.md](../research/2026-11-25-ch4-wind-datasets.md).

## Constraints

- **Disk.** The development Mac runs near full (as low as ~0.5 GB free on 2026-11-25, mostly from macOS swap). Runs
  refuse to start under 400 MB spare; EDGAR is reduced to 14 MB and its raw files deleted.
- **Rate budgets.** Open-Meteo's free tier is 600 calls/min, 5,000/h, 10,000/day per machine, and each grid point is a
  call. A global 5 degree run spends about 5,000 (ERA5 plus CAMS). One live collection at a time.
- **Licences.** Carbon Mapper data is non-commercial, share-alike and revocable; Open-Meteo's free API and Earth Engine
  are non-commercial. The repo is private and the work is research. Revisit before any public or commercial use.
