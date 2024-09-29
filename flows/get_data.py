from prefect import Flow, flow, get_run_logger, task
from surprise import Dataset

from deps.data import transform_data
from tasks.preprocess import preprocess_data
import pandas as pd
import os


@task
def load_data_csv(data_path: str = "../data/raw/ratings.csv"):
    return pd.read_csv(data_path, delimiter=";", encoding="ISO-8859-1")


@task
def preprocess_data(data: pd.DataFrame) -> Dataset:
    data = data.rename(
        columns={"User-ID": "user_id", "ISBN": "isbn", "Book-Rating": "book_rating"},
    )
    data = transform_data(data)
    return data


@flow(name="Train model", log_prints=True)
def get_data():
    data = load_data_csv()
    data = preprocess_data(data)
    return data


# Execute the flow
if __name__ == "__main__":
    get_data()
