import pandas as pd
from pandas import DataFrame
from surprise import Dataset, Reader


def read_csv_data(file_path="data/raw/BX-Book-Ratings.csv"):
    data = pd.read_csv(file_path, delimiter=";", encoding="ISO-8859-1")
    data.rename(
        columns={"User-ID": "user_id", "ISBN": "isbn", "Book-Rating": "book_rating"},
        inplace=True,
    )
    return data


def transform_data(data: DataFrame) -> Dataset:
    reader = Reader(rating_scale=(1, 10))
    data_formatted = Dataset.load_from_df(
        data[["user_id", "isbn", "book_rating"]], reader
    )
    return data_formatted
