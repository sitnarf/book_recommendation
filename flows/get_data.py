import pandas as pd
from pandas import DataFrame
from prefect import flow, task


@task
def load_data_csv(data_path: str = "./data/raw/ratings.csv"):
    return pd.read_csv(data_path, delimiter=";", encoding="ISO-8859-1")


@task
def preprocess_data(data: pd.DataFrame) -> DataFrame:
    data = data.rename(
        columns={"User-ID": "user_id", "ISBN": "isbn", "Book-Rating": "book_rating"},
    )
    return data


@flow(name="Get data", log_prints=True)
def get_data():
    data = load_data_csv()
    data = preprocess_data(data)
    return data


if __name__ == "__main__":
    get_data()
