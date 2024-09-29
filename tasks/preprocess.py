from prefect import task
import pandas as pd


@task
def preprocess_data(input_path: str, output_path: str):
    data = pd.read_csv(input_path, delimiter=";", encoding="ISO-8859-1")
    data.rename(
        columns={"User-ID": "user_id", "ISBN": "isbn", "Book-Rating": "book_rating"},
        inplace=True,
    )
    return data
