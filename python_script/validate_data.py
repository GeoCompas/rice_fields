import click
import pandas as pd
from glob import glob


@click.command(short_help="list files no annotations")
@click.option("--csv_folder", help="Input csv folder", type=str)
def run(csv_folder):
    all_files = glob(f"{csv_folder}/*.csv")

    def has_annotation(df_):
        columns = df_.columns.tolist()
        if not all([i in columns for i in ["y_ph", "y_fd"]]):
            return False
        if df_["y_ph"].dropna().empty:
            return False
        if df_["y_fd"].dropna().empty:
            return False
        return True

    all_df = {i: pd.read_csv(i) for i in all_files}
    for k, v in all_df.items():
        if not has_annotation(v):
            print(f"{k.split('/')[-1].split('_')[0]} has no annotation")


if __name__ == "__main__":
    run()
