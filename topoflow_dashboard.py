# topoflow_dashboard.py

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import xarray as xr
import numpy as np
import os

# -----------------------------
# File paths (override with environment variables)
# -----------------------------
DATA_DIR = os.environ.get("TOPOFLOW_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))

u_file = os.environ.get("TOPOFLOW_U_FILE", os.path.join(DATA_DIR, "10mucomponent.nc"))
v_file = os.environ.get("TOPOFLOW_V_FILE", os.path.join(DATA_DIR, "10mvcomponent.nc"))
methane_folder = os.environ.get("TOPOFLOW_METHANE_DIR", os.path.join(DATA_DIR, "methanedata"))

# -----------------------------
# Load ERA5 wind components
# -----------------------------
u_ds = xr.open_dataset(u_file)
v_ds = xr.open_dataset(v_file)

u_data = u_ds['u10']
v_data = v_ds['v10']

lats = u_ds['latitude'].values
lons = u_ds['longitude'].values
times = u_ds['valid_time'].values

# -----------------------------
# Load methane data
# -----------------------------
methane_data_dict = {}

for f in sorted(os.listdir(methane_folder)):
    if f.endswith(".nc"):
        ds = xr.open_dataset(os.path.join(methane_folder, f))
        da = ds['mtch4']
        date_str = os.path.splitext(f)[0]
        methane_data_dict[date_str] = da

# -----------------------------
# Stagnation points
# -----------------------------
# nodes, saddles, foci are Nx2 arrays: [[lon, lat], ...]

nodes = np.empty((0, 2))
saddles = np.empty((0, 2))
foci = np.empty((0, 2))

# -----------------------------
# Dash app
# -----------------------------
app = dash.Dash(__name__)

# Arial font theme
FONT = "Arial, sans-serif"

app.layout = html.Div(
    style={
        "fontFamily": FONT,
        "backgroundColor": "#f7f9fc",
        "padding": "25px",
        "minHeight": "100vh",
    },
    children=[
        html.H1(
            "TopoFlow: Wind + CH4 Interactive Map",
            style={
                "fontFamily": FONT,
                "fontWeight": "600",
                "fontSize": "30px",
                "color": "#1f2937",
                "textAlign": "center",
                "marginBottom": "8px",
            },
        ),

        html.P(
            "Global atmospheric wind vectors and methane distribution",
            style={
                "fontFamily": FONT,
                "fontSize": "15px",
                "color": "#6b7280",
                "textAlign": "center",
                "marginBottom": "25px",
            },
        ),

        dcc.Graph(
            id='wind-map',
            style={
                "height": "75vh",
                "backgroundColor": "white",
                "borderRadius": "12px",
            },
            config={
                "displayModeBar": True,
                "scrollZoom": True,
            }
        ),

        html.Div(
            style={
                "padding": "20px 30px",
                "backgroundColor": "white",
                "borderRadius": "12px",
                "marginTop": "15px",
                "boxShadow": "0 2px 8px rgba(0,0,0,0.04)",
            },
            children=[
                html.Label(
                    "Time",
                    style={
                        "fontFamily": FONT,
                        "fontWeight": "600",
                        "fontSize": "15px",
                        "color": "#374151",
                        "marginBottom": "10px",
                    },
                ),

                dcc.Slider(
                    id='time-slider',
                    min=0,
                    max=len(times)-1,
                    value=0,
                    marks={
                        i: {
                            "label": str(np.datetime64(times[i], 'D')),
                            "style": {
                                "fontFamily": FONT,
                                "fontSize": "11px",
                                "color": "#6b7280",
                            }
                        }
                        for i in range(
                            0,
                            len(times),
                            max(1, len(times)//10)
                        )
                    },
                    step=1,
                    tooltip={
                        "placement": "bottom",
                        "always_visible": False,
                    },
                ),
            ],
        ),
    ],
)

# -----------------------------
# Callback
# -----------------------------
@app.callback(
    Output('wind-map', 'figure'),
    [Input('time-slider', 'value')]
)
def update_figure(time_idx):

    # Extract wind data
    u_slice = u_data.isel(valid_time=time_idx).values
    v_slice = v_data.isel(valid_time=time_idx).values

    # Subsample for performance
    step = 20

    u_sub = u_slice[::step, ::step]
    v_sub = v_slice[::step, ::step]

    lat_sub = lats[::step]
    lon_sub = lons[::step]

    lon_grid, lat_grid = np.meshgrid(lon_sub, lat_sub)

    # -----------------------------
    # CH4 slice
    # -----------------------------
    date_str = str(
        np.datetime64(times[time_idx], 'D')
    ).replace('-', '')

    if date_str in methane_data_dict:
        ch4_slice = methane_data_dict[date_str].values[::step, ::step]
    else:
        ch4_slice = np.zeros_like(u_sub)

    # -----------------------------
    # Build figure
    # -----------------------------
    fig = go.Figure()

    # Background land/ocean approximation
    fig.add_trace(
        go.Heatmap(
            z=np.zeros_like(ch4_slice),
            x=lon_sub,
            y=lat_sub,
            colorscale=[
                [0, 'lightgreen'],
                [1, 'lightblue']
            ],
            showscale=False,
            opacity=0.2,
        )
    )

    # -----------------------------
    # CH4 heatmap
    # -----------------------------
    fig.add_trace(
        go.Heatmap(
            z=ch4_slice,
            x=lon_sub,
            y=lat_sub,
            colorscale='Oranges',
            opacity=0.6,
            colorbar=dict(
                title=dict(
                    text="CH4 (ppb)",
                    font=dict(
                        family=FONT,
                        size=14,
                        color="#374151"
                    ),
                ),
                tickfont=dict(
                    family=FONT,
                    size=12,
                    color="#4b5563"
                ),
            ),
        )
    )

    # -----------------------------
    # Wind vectors
    # -----------------------------
    for i in range(lat_grid.shape[0]):
        for j in range(lat_grid.shape[1]):

            fig.add_trace(
                go.Scatter(
                    x=[
                        lon_grid[i, j],
                        lon_grid[i, j] + u_sub[i, j] * 0.1
                    ],
                    y=[
                        lat_grid[i, j],
                        lat_grid[i, j] + v_sub[i, j] * 0.1
                    ],
                    mode='lines',
                    line=dict(
                        color='blue',
                        width=1
                    ),
                    showlegend=False,
                )
            )

    # -----------------------------
    # Stagnation points
    # -----------------------------
    if nodes.shape[0] > 0:
        fig.add_trace(
            go.Scatter(
                x=nodes[:, 0],
                y=nodes[:, 1],
                mode='markers',
                marker=dict(
                    color='red',
                    size=6
                ),
                name='Node'
            )
        )

    if saddles.shape[0] > 0:
        fig.add_trace(
            go.Scatter(
                x=saddles[:, 0],
                y=saddles[:, 1],
                mode='markers',
                marker=dict(
                    color='blue',
                    size=6
                ),
                name='Saddle'
            )
        )

    if foci.shape[0] > 0:
        fig.add_trace(
            go.Scatter(
                x=foci[:, 0],
                y=foci[:, 1],
                mode='markers',
                marker=dict(
                    color='green',
                    size=6
                ),
                name='Focus'
            )
        )

    # -----------------------------
    # Layout / formatting
    # -----------------------------
    fig.update_layout(
        title=dict(
            text=f"ERA5 Wind + CH4 ({np.datetime64(times[time_idx], 'D')})",
            font=dict(
                family=FONT,
                size=20,
                color="#1f2937"
            ),
            x=0.5,
            xanchor="center",
        ),

        font=dict(
            family=FONT,
            size=13,
            color="#374151"
        ),

        xaxis=dict(
            title=dict(
                text="Longitude",
                font=dict(
                    family=FONT,
                    size=14,
                    color="#374151"
                ),
            ),
            range=[-180, 180],
            tickfont=dict(
                family=FONT,
                size=11,
                color="#6b7280"
            ),
            gridcolor="#e5e7eb",
            zerolinecolor="#d1d5db",
        ),

        yaxis=dict(
            title=dict(
                text="Latitude",
                font=dict(
                    family=FONT,
                    size=14,
                    color="#374151"
                ),
            ),
            range=[-90, 90],
            scaleanchor='x',
            tickfont=dict(
                family=FONT,
                size=11,
                color="#6b7280"
            ),
            gridcolor="#e5e7eb",
            zerolinecolor="#d1d5db",
        ),

        plot_bgcolor="white",
        paper_bgcolor="white",

        margin=dict(
            l=60,
            r=65,
            t=65,
            b=50
        ),

        hoverlabel=dict(
            font=dict(
                family=FONT,
                size=13
            )
        ),

        legend=dict(
            font=dict(
                family=FONT,
                size=12
            ),
            bgcolor="rgba(255,255,255,0.8)",
        ),
    )

    return fig

# -----------------------------
# Run app
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
