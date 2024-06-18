import dash
from dash import Dash, html, dcc, Input, Output, State, callback_context
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import os
from glob import glob


def adjust_doy_column(doy_column):
    """
    Adjust the day of year (doy) column to ensure correct representation across multiple years.

    Parameters:
    - doy_column: Array or list of day of year (doy) values.

    Returns:
    - Adjusted array or list of day of year (doy) values.
    """
    # Find the index where the values switch from the first range (5 to 350) to the second range (5 to 50)
    switch_index = np.argmax(np.diff(doy_column) < 0) + 1

    # Adjust values after the switch to account for the transition to the next year
    adjusted_doy_column = np.array(doy_column)
    adjusted_doy_column[switch_index:] += 365

    return adjusted_doy_column


def render_spectral_info_on_field(csv):
    df = pd.read_csv(csv)
    if "doy" not in df.columns:
        raise ValueError("The DataFrame must contain a 'doy' column.")
    else:
        df["doy"] = adjust_doy_column(df["doy"])
    field_id = csv.split("/")[-1].split("_")[0]
    return df, field_id, csv


data_pth = os.getenv("DATA_PATH", "data")

files_annotated = [
    f.replace("output", "input")
    for f in glob(f"{data_pth}/output/**/*.csv", recursive=True)
]
print("Files annotated: ", len(files_annotated))

csvs = [
    f
    for f in glob(f"{data_pth}/input/**/*.csv", recursive=True)
    if f not in files_annotated
]

app = Dash(__name__)

# Simple layout with a Plotly graph and buttons to confirm selections
app.layout = html.Div(
    [
        html.Button("Save Annotations and Next", id="confirm-btn", n_clicks=0),
        dcc.Graph(
            id="ndvi-time-series",
            config={"staticPlot": False, "scrollZoom": True},
            clear_on_unhover=True,
        ),
        html.Div(id="selected-period-output"),  # To display selected period
        dcc.Store(
            id="field-index", data=0
        ),  # To keep track of which field is being shown
        dcc.Store(id="ndvi-data-store"),
        dcc.Store(id="tooltip-visibility", data={"visible": False}),
        dcc.Tooltip(id="image_tooltip", direction="bottom"),
        dcc.Interval(
            id="tooltip-interval",
            interval=5000,  # in milliseconds
            n_intervals=0,
            max_intervals=1,  # This makes sure it only runs once after activation
        ),
        html.Button("Mark Cropping Window", id="mark-cropping-window-btn", n_clicks=0),
        html.Button("Mark Flooding Window", id="mark-flooding-window-btn", n_clicks=0),
        dcc.Store(
            id="annotations-store",
            data={"cropping_windows": [], "flooding_windows": []},
        ),
    ],
    style={"width": "100%"},
)


@app.callback(
    Output("ndvi-time-series", "figure"),
    [Input("field-index", "data"), Input("annotations-store", "data")],
    State("ndvi-data-store", "data"),
)
def update_graph(field_index, annotations, ndvi_data_store):
    if field_index >= len(csvs):
        return go.Figure()  # Return an empty figure if no more CSVs are left

    csv = csvs[field_index]
    df, field_id, folder_id = render_spectral_info_on_field(csv)
    df["s2_ndvi_smoothed"] = savgol_filter(df["s2_ndvi"], 10, 3)
    df["s1_vh_smoothed"] = savgol_filter(df["s1_vh"], 10, 3)
    df["s1_vv_smoothed"] = savgol_filter(df["s1_vv"], 10, 3)
    df["s2_ndwi_smoothed"] = savgol_filter(df["s2_ndwi"], 10, 3)
    df["s2_mndwi_smoothed"] = savgol_filter(df["s2_mndwi"], 10, 3)

    fig = go.Figure(
        data=[  # go.Scatter(x=df['doy'], y=df['s2_ndwi_smoothed']*15, mode='lines+markers', name='S2 NDWI Smoothed', line=dict(color='blue')),
            go.Scatter(
                x=df["doy"],
                y=df["s2_mndwi_smoothed"] * 15,
                mode="lines+markers",
                name="S2 MNDWI Smoothed",
                line=dict(color="red"),
            ),
            go.Scatter(
                x=df["doy"],
                y=df["s2_ndvi_smoothed"] * 15,
                mode="lines+markers",
                name="S2 NDVI Smoothed",
                line=dict(color="green"),
            ),
            go.Scatter(
                x=df["doy"],
                y=df["s1_vh_smoothed"],
                mode="lines+markers",
                name="S1 VH Smoothed",
                line=dict(color="orange"),
            ),
            go.Scatter(
                x=df["doy"],
                y=df["s1_vv_smoothed"],
                mode="lines+markers",
                name="S1 VV Smoothed",
                line=dict(color="purple"),
            ),
        ]
    )
    title_text = f"Field {field_id}"

    fig.update_layout(
        title=title_text,
        xaxis_title="DOY",
        yaxis_title="Smoothed Values",
        dragmode="select",
        height=800,
        xaxis=dict(fixedrange=False, tickmode="auto", gridcolor="LightGrey"),
        yaxis=dict(fixedrange=True),
    )

    # Logic to add planting and harvest windows to the figure
    for window in annotations.get("cropping_windows", []):
        fig.add_vrect(
            x0=int(window["start"]),
            x1=int(window["end"]),
            fillcolor="green",
            opacity=0.15,
        )
        fig.add_shape(
            type="line",
            x0=int(window["start"]),
            y0=-3,
            x1=int(window["end"]),
            y1=-3,
            line=dict(color="green", width=1),
        )
        # Calculate the midpoint for the text annotation
        mid_point = (window["start"] + window["end"]) / 2

        # Adding text annotation for the duration of the cropping window
        fig.add_annotation(
            x=mid_point,
            y=-3,
            text=f"{int(window['end'] - window['start'])} days",
            showarrow=False,
            font=dict(family="Arial", size=15, color="white"),
            align="center",
            bgcolor="green",
            opacity=1,
        )

    # Logic to add flooding windows to the figure
    for window in annotations.get("flooding_windows", []):
        fig.add_vrect(
            x0=int(window["start"]),
            x1=int(window["end"]),
            fillcolor="blue",
            opacity=0.15,
        )
        fig.add_shape(
            type="line",
            x0=int(window["start"]),
            y0=-2,
            x1=int(window["end"]),
            y1=-2,
            line=dict(color="blue", width=1),
        )
        # Calculate the midpoint for the text annotation
        mid_point = (window["start"] + window["end"]) / 2

        # Adding text annotation for the duration of the cropping window
        fig.add_annotation(
            x=mid_point,
            y=-2,
            text=f"{int(window['end'] - window['start'])} days",
            showarrow=False,
            font=dict(family="Arial", size=15, color="white"),
            align="center",
            bgcolor="blue",
            opacity=1,
        )

    return fig


@app.callback(
    [
        Output("selected-period-output", "children", allow_duplicate=True),
        Output("annotations-store", "data", allow_duplicate=True),
        Output("field-index", "data"),
    ],
    [Input("confirm-btn", "n_clicks")],
    [State("annotations-store", "data"), State("field-index", "data")],
    prevent_initial_call=True,
)
def save_annotations_and_next(n_clicks, annotations, field_index):
    if field_index < len(csvs):
        # Load the data for the current field
        csv = csvs[field_index]
        df, field_id, csv_path = render_spectral_info_on_field(csv)

        # Ensure 'doy', 'y_ph', and 'y_fd' columns are present
        if "doy" not in df.columns:
            raise ValueError("The DataFrame must contain a 'doy' column.")

        # Initialize y_ph and y_fd with NaN if they don't already exist
        if "y_ph" not in df.columns:
            df["y_ph"] = np.nan
        if "y_fd" not in df.columns:
            df["y_fd"] = np.nan

        print(f"Processing annotations for field {field_id}")

        # Helper function to find the closest index
        def find_closest_index(value, array):
            return (np.abs(array - value)).argmin()

        # Process each cropping window for y_ph
        for window in annotations["cropping_windows"]:
            window["planting_date_0"] = window["start"]
            window["harvest_date_0"] = window["end"]

            # Find the closest start and end indices
            start_idx = find_closest_index(window["start"], df["doy"].values)
            end_idx = find_closest_index(window["end"], df["doy"].values)

            # Ensure the indices are in the correct order
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx

            y_ph = np.linspace(1.0, 0.0, end_idx - start_idx + 1)

            # Update the DataFrame without overwriting previous values
            df.loc[start_idx:end_idx, "y_ph"] = y_ph
            print(
                f"Updated y_ph from {df['doy'].iloc[start_idx]} to {df['doy'].iloc[end_idx]}"
            )

        # Process each flooding window for y_fd
        for window in annotations["flooding_windows"]:
            # Find the closest start and end indices
            start_idx = find_closest_index(window["start"], df["doy"].values)
            end_idx = find_closest_index(window["end"], df["doy"].values)

            # Ensure the indices are in the correct order
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx

            y_fd = np.linspace(1.0, 0.0, end_idx - start_idx + 1)

            # Update the DataFrame without overwriting previous values
            df.loc[start_idx:end_idx, "y_fd"] = y_fd
            print(
                f"Updated y_fd from {df['doy'].iloc[start_idx]} to {df['doy'].iloc[end_idx]}"
            )

        os.makedirs(os.path.dirname(csv_path.replace("input", "output")), exist_ok=True)
        df.to_csv(csv_path.replace("input", "output"), index=False)
        print(f"Annotations for field {field_id} saved to CSV")
        print(f"***" * 20, "\n")

    # Move to the next field
    next_field_index = field_index + 1
    return (
        f"Annotations saved for field {field_id}!",
        {"cropping_windows": [], "flooding_windows": []},
        next_field_index,
    )


@app.callback(
    Output("annotations-store", "data"),
    [
        Input("mark-cropping-window-btn", "n_clicks"),
        Input("mark-flooding-window-btn", "n_clicks"),
    ],
    [
        State("ndvi-time-series", "selectedData"),
        State("annotations-store", "data"),
        State("field-index", "data"),
    ],
    prevent_initial_call=True,
)
def register_window(
    cropping_clicks, flooding_clicks, selectedData, annotations, field_index
):
    ctx = callback_context

    if ctx.triggered and selectedData and selectedData["range"]["x"]:
        start_date, end_date = selectedData["range"]["x"]
        window = {"start": start_date, "end": end_date}
        day_difference = int(end_date) - int(start_date)

        # period_output = f"Selected period: {day_difference} days"

        if ctx.triggered[0]["prop_id"] == "mark-cropping-window-btn.n_clicks":
            # Append the new window to the list of cropping windows
            annotations["cropping_windows"].append(
                window
            )  # annotations["cropping_duration"] = day_difference  # print(f"Added cropping window: {annotations}")
        elif ctx.triggered[0]["prop_id"] == "mark-flooding-window-btn.n_clicks":
            # Append the new window to the list of flooding windows
            annotations["flooding_windows"].append(
                window
            )  # annotations["flooding_duration"] = day_difference  # print(f"Added flooding window: {annotations}")

    return annotations


@app.callback(
    [
        Output("image_tooltip", "show"),
        Output("image_tooltip", "bbox"),
        Output("image_tooltip", "children", allow_duplicate=True),
        Output("tooltip-interval", "n_intervals"),
    ],
    [Input("ndvi-time-series", "clickData"), Input("tooltip-interval", "n_intervals")],
    State("field-index", "data"),
    prevent_initial_call=True,
)
def display_click(clickData, n_intervals, field_index):
    ctx = callback_context

    if not ctx.triggered:
        return [False, {}, None, dash.no_update]

    if ctx.triggered[0]["prop_id"] == "ndvi-time-series.clickData" and clickData:
        click_point = clickData["points"][0]
        date_clicked = click_point["x"]
        bbox = {
            "left": click_point["x"],
            "top": click_point["y"],
            "width": 100,
            "height": 50,
        }
        children = f"Clicked date: {date_clicked}"
        return [True, bbox, children, 0]
    elif ctx.triggered[0]["prop_id"] == "tooltip-interval":
        # Timeout occurred, hide tooltip
        return [False, {}, None, dash.no_update]

    # Return a default tuple to avoid returning None
    return [False, {}, None, dash.no_update]


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8060)
