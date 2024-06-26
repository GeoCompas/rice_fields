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
    max_value = doy_column.max()
    adjusted_doy_column = np.array(doy_column)

    if max_value < 366:
        # Adjust values after the switch to account for the transition to the next year
        adjusted_doy_column[switch_index:] += 365

    return adjusted_doy_column


def ydict2windows(y_list, window_type):
    windows = []
    try:
        start_date = None
        for i, item in enumerate(y_list):
            doy = item["doy"]
            y_ph = item["val"]

            if not pd.isna(y_ph) and y_ph != 0:
                if start_date is None:
                    start_date = doy
            else:
                if start_date is not None:
                    end_date = y_list[i]["doy"]
                    windows.append(
                        {"start": start_date, "end": end_date, "type": window_type}
                    )
                    start_date = None

        # check windows close
        if start_date is not None:
            end_date = y_list[-1]["doy"]
            windows.append({"start": start_date, "end": end_date, "type": window_type})
    except Exception as ex:
        print(ex)
    return windows


def read_csv(csv_path):
    # Load the data for the current field
    has_error = False
    annotations = []
    df: pd.DataFrame = pd.read_csv(csv_path)
    # get old annotations
    if "doy" not in df.columns:
        has_error = True
        print(csv_path, "does not contain doy column")
    df["doy"] = adjust_doy_column(df["doy"])
    df.drop_duplicates(subset=["doy"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    # get old annotations
    if "y_ph" in df.columns and not has_error:
        df_ph = df[["doy", "y_ph"]].copy()
        df_ph["val"] = df_ph["y_ph"]
        annotations += ydict2windows(
            df_ph.to_dict(orient="records"), "cropping_windows"
        )
    if "y_fd" in df.columns and not has_error:
        df_fd = df[["doy", "y_fd"]].copy()
        df_fd["val"] = df_fd["y_fd"]
        annotations += ydict2windows(
            df_fd.to_dict(orient="records"), "flooding_windows"
        )

    # reset file
    df["y_ph"] = np.nan
    df["y_fd"] = np.nan
    output = {"annotations": annotations, "data": df.copy(), "has_error": has_error}

    return output


# Helper function to find the closest index
def find_closest_index(value, array):
    return (np.abs(array - value)).argmin()


## ==================
## APP
## ==================

data_pth = os.getenv("DATA_PATH", "data")

files_annotated = [
    f.replace("/output/", "/input/") for f in glob(f"{data_pth}/output/**/*.csv")
]
files_incomplete = [
    f.replace("/incomplete/", "/input/")
    for f in glob(f"{data_pth}/incomplete/**/*.csv")
]
all_csv = sorted([f for f in glob(f"{data_pth}/input/**/*.csv")])

csvs_filter = [
    {
        "file_path": f,
        "folder_id": f.split("/")[-2],
        "field_id": f.split("/")[-1].split("_")[0],
        **read_csv(f),
    }
    for f in all_csv
    if f not in [*files_annotated, *files_incomplete]
]
# filter error csv
csvs = [item for item in csvs_filter if not item.get("has_error")]

print("=" * 20)
print("Files input: ", len(all_csv))
print("Files annotated: ", len(files_annotated))
print("Files incomplete: ", len(files_incomplete))
print("Files filter: ", len(csvs_filter))
print("Files no error: ", len(csvs))
print("=" * 20)

ALL_CSV_COUNT = len(csvs)

app = Dash(__name__)
btn_style = {
    "fontSize": "15px",
    "marginLeft": "5px",
    "marginRight": "5px",
    "padding": "3px 8px",
    "borderRadius": "5px",
}
btn_style_lg = {
    **btn_style,
    "padding": "5px",
    "fontSize": "18px",
    "border": "4px solid",
}

# Simple layout with a Plotly graph and buttons to confirm selections
app.layout = html.Div(
    [
        html.Button(
            "Mark Window",
            id="mark-window-btn",
            n_clicks=0,
            style={**btn_style_lg, "borderColor": "gray"},
        ),
        html.Button(
            " -- Save and Next --",
            id="confirm-btn",
            n_clicks=0,
            style={**btn_style_lg, "marginLeft": "30px"},
        ),
        dcc.Graph(
            id="ndvi-time-series",
            config={"staticPlot": False, "scrollZoom": True},
            clear_on_unhover=True,
        ),
        html.Div(id="selected-period-output"),
        # To display selected period
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
        html.Button("Prev file", id="prev-btn", n_clicks=0, style=btn_style),
        html.Button("Next file", id="next-btn", n_clicks=0, style=btn_style),
        html.Button(
            "Save file with incomplete data ",
            id="incomplete-btn",
            n_clicks=0,
            style=btn_style,
        ),
        html.Button(
            "remove last window ",
            id="rm-last-window-btn",
            n_clicks=0,
            style=btn_style,
        ),
    ],
    style={"width": "100%", "textAlign": "center"},
)


@app.callback(
    Output("ndvi-time-series", "figure"),
    [Input("field-index", "data")],
    State("ndvi-data-store", "data"),
)
def update_graph(field_index, ndvi_data_store):
    if field_index >= ALL_CSV_COUNT:
        return go.Figure()  # Return an empty figure if no more CSVs are left

    csv = csvs[field_index]
    df_: pd.DataFrame = csv.get("data")
    df = df_.copy()
    field_id = csv.get("field_id")
    folder_id = csv.get("folder_id")
    annotations = csv.get("annotations", [])

    df["s2_ndvi_smoothed"] = savgol_filter(df["s2_ndvi"], 10, 3)
    df["s1_vh_smoothed"] = savgol_filter(df["s1_vh"], 10, 3)
    df["s1_vv_smoothed"] = savgol_filter(df["s1_vv"], 10, 3)
    df["s2_ndwi_smoothed"] = savgol_filter(df["s2_ndwi"], 10, 3)
    df["s2_mndwi_smoothed"] = savgol_filter(df["s2_mndwi"], 10, 3)

    fig = go.Figure(
        data=[
            go.Scatter(
                x=df["doy"],
                y=df["s2_mndwi_smoothed"] * 15,
                mode="lines+markers",
                name="S2 MNDWI Smoothed",
                line=dict(color="red"),
                opacity=0.7,  # opacity
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
                y=df["s2_ndwi_smoothed"] * 15,
                mode="lines+markers",
                name="S2 NDWI Smoothed",
                line=dict(color="blue"),
                opacity=0.3,  # opacity
            ),
            go.Scatter(
                x=df["doy"],
                y=df["s1_vh_smoothed"],
                mode="lines+markers",
                name="S1 VH Smoothed",
                line=dict(color="orange"),
                # opacity=0.5,
            ),
            go.Scatter(
                x=df["doy"],
                y=df["s1_vv_smoothed"],
                mode="lines+markers",
                name="S1 VV Smoothed",
                line=dict(color="purple"),
                # opacity=0.5,
            ),
        ]
    )
    fig.update_traces(marker_size=8)
    title_text = (
        f"Field:\t  {folder_id}/{field_id} ---> ({field_index + 1} / {ALL_CSV_COUNT})"
    )

    fig.update_layout(
        title=title_text,
        xaxis_title="DOY",
        yaxis_title="Smoothed Values",
        dragmode="select",
        height=700,
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
    for k, window in enumerate(annotations):
        # use the doy most near in dataset
        start_idx = find_closest_index(window["start"], df["doy"].values)
        end_idx = find_closest_index(window["end"], df["doy"].values)

        x0 = df["doy"][start_idx]
        x1 = df["doy"][end_idx]

        last_period = (x1 - x0) // 4
        mid_point = (window["start"] + window["end"]) / 2

        type_windows = window.get("type")
        color = "green" if type_windows == "cropping_windows" else "blue"
        y_01 = -25 if type_windows == "cropping_windows" else -28
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=color,
            opacity=0.15,
        )
        if type_windows == "cropping_windows":
            # second crop
            fig.add_vrect(
                x0=x1 - last_period,
                x1=x1,
                fillcolor="green",
                opacity=0.15,
            )

        fig.add_shape(
            type="line",
            x0=x0,
            y0=y_01,
            x1=x1,
            y1=y_01,
            line=dict(color=color, width=1),
        )
        fig.add_annotation(
            x=mid_point,
            y=y_01,
            text=f"{int(x1 - x0)} days",
            showarrow=False,
            font=dict(family="Arial", size=12, color="white"),
            align="center",
            bgcolor=color,
            opacity=1,
        )
        fig.add_annotation(
            x=mid_point,
            y=10 if type_windows == "cropping_windows" else 12,
            text=f"RM-{k}",
            showarrow=False,
            font=dict(family="Arial", size=12, color="white"),
            align="center",
            bgcolor=color,
            opacity=0.7,
            captureevents=True,
        )

    return fig


@app.callback(
    [
        Output("selected-period-output", "children", allow_duplicate=True),
        Output("field-index", "data"),
    ],
    [Input("confirm-btn", "n_clicks")],
    [State("field-index", "data")],
    prevent_initial_call=True,
)
def save_annotations_and_next(n_clicks, field_index):
    field_id = "---"
    next_field_index = field_index
    if field_index < ALL_CSV_COUNT:
        csv = csvs[field_index]

        df: pd.DataFrame = csv.get("data")
        field_id = csv.get("field_id")
        file_path = csv.get("file_path")
        annotations = csv.get("annotations", [])

        print(f"Processing annotations for field {field_id}")

        cropping_windows = [
            i for i in annotations if i.get("type") == "cropping_windows"
        ]
        flooding_windows = [
            i for i in annotations if i.get("type") == "flooding_windows"
        ]

        # Process each cropping window for y_ph
        for window in cropping_windows:
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
        for window in flooding_windows:
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

        os.makedirs(
            os.path.dirname(file_path.replace("input", "output")), exist_ok=True
        )
        df.to_csv(file_path.replace("/input/", "/output/"), index=False)
        print(f"Annotations for field {field_id} saved to CSV")
        print(f"***" * 10, "\n")

        next_field_index = field_index + 1
    return (
        f"Annotations saved for field {field_id}!",
        next_field_index,
    )


@app.callback(
    [
        Output("selected-period-output", "children", allow_duplicate=True),
        Output("field-index", "data", allow_duplicate=True),
    ],
    [Input("next-btn", "n_clicks")],
    [State("field-index", "data")],
    prevent_initial_call=True,
)
def next_file(n_clicks, field_index):
    field_id = "---"
    next_field_index = field_index
    if field_index < ALL_CSV_COUNT:
        csv = csvs[field_index]
        field_id = csv.get("field_id")

        print(f"next field  {field_id}")
        print(f"***" * 10, "\n")
        next_field_index = field_index + 1

    return (
        f"Prev field {field_id}!",
        next_field_index,
    )


@app.callback(
    [
        Output("selected-period-output", "children", allow_duplicate=True),
        Output("field-index", "data", allow_duplicate=True),
    ],
    [Input("incomplete-btn", "n_clicks")],
    [State("field-index", "data")],
    prevent_initial_call=True,
)
def incomplete_file(n_clicks, field_index):
    field_id = "--"
    next_field_index = field_index
    if field_index < ALL_CSV_COUNT:
        csv = csvs[field_index]

        df: pd.DataFrame = csv.get("data")
        field_id = csv.get("field_id")
        file_path = csv.get("file_path")

        os.makedirs(
            os.path.dirname(file_path.replace("input", "incomplete")), exist_ok=True
        )
        df.to_csv(file_path.replace("/input/", "/incomplete/"), index=False)
        print(f"Incomplete field {field_id} saved to CSV")
        print(f"***" * 10, "\n")

        next_field_index = field_index + 1

    return (
        f"incomplete field {field_id}!",
        next_field_index,
    )


@app.callback(
    [
        Output("selected-period-output", "children", allow_duplicate=True),
        Output("field-index", "data", allow_duplicate=True),
    ],
    [Input("prev-btn", "n_clicks")],
    [State("field-index", "data")],
    prevent_initial_call=True,
)
def prev_file(n_clicks, field_index):
    field_id = "--"
    next_field_index = field_index
    if field_index > ALL_CSV_COUNT:
        field_index = ALL_CSV_COUNT
    if field_index > 0:
        csv = csvs[field_index - 1]

        field_id = csv.get("field_id")

        print(f"prev field  {field_id}")
        print(f"***" * 10, "\n")

        next_field_index -= 1
    return (
        f"Prev field {field_id}!",
        next_field_index,
    )


@app.callback(
    Output("field-index", "data", allow_duplicate=True),
    [
        Input("mark-window-btn", "n_clicks"),
    ],
    [
        State("ndvi-time-series", "selectedData"),
        State("field-index", "data"),
    ],
    prevent_initial_call=True,
)
def register_window(window_clicks, selectedData, field_index):
    ctx = callback_context
    try:
        if ctx.triggered and selectedData and selectedData["range"]["x"]:
            start_date, end_date = selectedData["range"]["x"]
            window = {"start": start_date, "end": end_date}
            csv = csvs[field_index]
            type_ = "cropping_windows"
            if (end_date - start_date) <= 60:
                type_ = "flooding_windows"
            window["type"] = type_
            csv["annotations"].append(window)
    except Exception as ex:
        print(ex)
    return field_index


@app.callback(
    Output("field-index", "data", allow_duplicate=True),
    [
        Input("rm-last-window-btn", "n_clicks"),
    ],
    [
        State("field-index", "data"),
    ],
    prevent_initial_call=True,
)
def remove_last_window(n_clicks, field_index):
    if n_clicks is None:
        return dash.no_update
    csv = csvs[field_index]

    if len(csv["annotations"]) > 0:
        csv["annotations"].pop()

    return field_index


@app.callback(
    Output("field-index", "data", allow_duplicate=True),
    [Input("ndvi-time-series", "clickAnnotationData")],
    [State("field-index", "data")],
    prevent_initial_call=True,
)
def display_double_click_data(clickAnnotationData, field_index):
    if clickAnnotationData is not None:
        try:
            text = clickAnnotationData["annotation"]["text"]
            index_annotation = int(text.split("-")[1])
            csv = csvs[field_index]
            csv["annotations"].pop(index_annotation)
        except Exception as ex:
            print(ex)
    return field_index


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8060)
