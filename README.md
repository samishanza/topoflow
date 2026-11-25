# TopoFlow

An interactive Dash dashboard that shows ERA5 10 m wind vectors over a methane (CH4) concentration map, with a time slider to step through each day. It also has placeholders for marking stagnation points in the wind field (nodes, saddles and foci).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The data files aren't included in this repo. Put them in a `data/` folder:

```
data/
├── 10mucomponent.nc      # ERA5 10 m U-wind (variable: u10)
├── 10mvcomponent.nc      # ERA5 10 m V-wind (variable: v10)
└── methanedata/
    └── YYYYMMDD.nc       # daily CH4 field (variable: mtch4), one file per day
```

You can also point to files somewhere else with environment variables:

| Variable | Default |
|---|---|
| `TOPOFLOW_DATA_DIR` | `./data` |
| `TOPOFLOW_U_FILE` | `$TOPOFLOW_DATA_DIR/10mucomponent.nc` |
| `TOPOFLOW_V_FILE` | `$TOPOFLOW_DATA_DIR/10mvcomponent.nc` |
| `TOPOFLOW_METHANE_DIR` | `$TOPOFLOW_DATA_DIR/methanedata` |

## Run

```bash
python topoflow_dashboard.py
```

Then open http://127.0.0.1:8050 in your browser.
