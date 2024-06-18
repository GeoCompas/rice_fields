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
files_incomplete = [
    f.replace("incomplete", "input")
    for f in glob(f"{data_pth}/incomplete/**/*.csv", recursive=True)
]
print("Files annotated: ", len(files_annotated))
print("Files incomplete: ", len(files_incomplete))

csvs = [
    f
    for f in glob(f"{data_pth}/input/**/*.csv", recursive=True)
    if f not in [*files_annotated,*files_incomplete]
]

app = Dash(__name__)
btn_style = {"marginLeft": "5px", "marginRight": "5px", "padding": "3px"}
# Simple layout with a Plotly graph and buttons to confirm selections
app.layout = html.Div(
    [
        html.Button("Prev file", id="prev-btn", n_clicks=0, style=btn_style),
        html.Button(
            " -- Save Annotations and Next --",
            id="confirm-btn",
            n_clicks=0,
            style=btn_style,
        ),
        html.Button("Next file", id="next-btn", n_clicks=0, style=btn_style),
        html.Button(
            "Save file with incomplete data ",
            id="incomplete-btn",
            n_clicks=0,
            style=btn_style,
        ),
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
        html.Button(
            "Mark Cropping Window",
            id="mark-cropping-window-btn",
            n_clicks=0,
            style=btn_style,
        ),
        html.Button(
            "Mark Flooding Window",
            id="mark-flooding-window-btn",
            n_clicks=0,
            style=btn_style,
        ),
        html.Button(
            "-- remove last annotation--",
            id="rm-last-window-btn",
            n_clicks=0,
            style=btn_style,
        ),
        dcc.Store(
            id="annotations-store",
            data={
                "cropping_windows": [],
                "flooding_windows": [],
                "annotation_type": [],
            },
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
    title_text = f"Field {field_id} - ({field_index})"

    fig.update_layout(
        title=title_text,
        xaxis_title="DOY",
        yaxis_title="Smoothed Values",
        dragmode="select",
        height=800,
        xaxis=dict(fixedrange=False, tickmode="auto", gridcolor="LightGrey"),
        yaxis=dict(fixedrange=True),
    )
    # default 1
    fig.add_shape(
        type="line",
        x0=0,
        y0=1,
        x1=400,
        y1=1,
        line=dict(color="black", width=0.3),
    )

    # Logic to add planting and harvest windows to the figure
    for window in annotations.get("cropping_windows", []):
        fig.add_vrect(
            x0=int(window["start"]),
            x1=int(window["end"]),
            fillcolor="green",
            opacity=0.15,
        )
        # second crop
        last_period = (int(window["end"]) - int(window["start"])) // 4
        fig.add_vrect(
            x0=int(window["end"]) - last_period,
            x1=int(window["end"]),
            fillcolor="green",
            opacity=0.15,
        )
        fig.add_shape(
            type="line",
            x0=int(window["start"]),
            y0=-5,
            x1=int(window["end"]),
            y1=-5,
            line=dict(color="green", width=1),
        )
        # Calculate the midpoint for the text annotation
        mid_point = (window["start"] + window["end"]) / 2

        # Adding text annotation for the duration of the cropping window
        fig.add_annotation(
            x=mid_point,
            y=-5,
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
        {"cropping_windows": [], "flooding_windows": [], "annotation_type": []},
        next_field_index,
    )


@app.callback(
    [
        Output("selected-period-output", "children", allow_duplicate=True),
        Output("annotations-store", "data", allow_duplicate=True),
        Output("field-index", "data", allow_duplicate=True),
    ],
    [Input("next-btn", "n_clicks")],
    [State("annotations-store", "data"), State("field-index", "data")],
    prevent_initial_call=True,
)
def next_file(n_clicks, annotations, field_index):
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

        print(f"next field  {field_id}")
        print(f"***" * 20, "\n")

    # Move to the next field
    next_field_index = field_index + 1
    return (
        f"Prev field {field_id}!",
        {"cropping_windows": [], "flooding_windows": [], "annotation_type": []},
        next_field_index,
    )

@app.callback(
    [
        Output("selected-period-output", "children", allow_duplicate=True),
        Output("annotations-store", "data", allow_duplicate=True),
        Output("field-index", "data", allow_duplicate=True),
    ],
    [Input("incomplete-btn", "n_clicks")],
    [State("annotations-store", "data"), State("field-index", "data")],
    prevent_initial_call=True,
)
def incomplete_file(n_clicks, annotations, field_index):
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

        os.makedirs(os.path.dirname(csv_path.replace("input", "incomplete")), exist_ok=True)
        df.to_csv(csv_path.replace("input", "incomplete"), index=False)
        print(f"Incomplete field {field_id} saved to CSV")
        print(f"***" * 20, "\n")

    # Move to the next field
    next_field_index = field_index + 1

    return (
        f"incomplete field {field_id}!",
        {"cropping_windows": [], "flooding_windows": [], "annotation_type": []},
        next_field_index,
    )


@app.callback(
    [
        Output("selected-period-output", "children", allow_duplicate=True),
        Output("annotations-store", "data", allow_duplicate=True),
        Output("field-index", "data", allow_duplicate=True),
    ],
    [Input("prev-btn", "n_clicks")],
    [State("annotations-store", "data"), State("field-index", "data")],
    prevent_initial_call=True,
)
def prev_file(n_clicks, annotations, field_index):
    if field_index < len(csvs) and field_index > 0:
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

        print(f"prev field  {field_id}")
        print(f"***" * 20, "\n")

        # Move to the next field
    next_field_index = field_index - 1
    return (
        f"Prev field {field_id}!",
        {"cropping_windows": [], "flooding_windows": [], "annotation_type": []},
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
            annotations["cropping_windows"].append(window)
            annotations["annotation_type"].append("cropping_windows")
        elif ctx.triggered[0]["prop_id"] == "mark-flooding-window-btn.n_clicks":
            annotations["flooding_windows"].append(window)
            annotations["annotation_type"].append("flooding_windows")
    return annotations


@app.callback(
    Output("annotations-store", "data", allow_duplicate=True),
    [
        Input("rm-last-window-btn", "n_clicks"),
    ],
    [
        State("annotations-store", "data"),
    ],
    prevent_initial_call=True,
)
def remove_last_window(n_clicks, annotations):
    if n_clicks is None:
        return dash.no_update

    annotation_type = annotations.get("annotation_type")
    if annotation_type and len(annotation_type):
        last_annotation = annotation_type[-1]
        annotations[last_annotation].pop()
        annotations["annotation_type"].pop()

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
