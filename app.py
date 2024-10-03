import pandas as pd
from flask import Flask
import mlflow
from pandas import Series

from flows.get_data import get_data

mlflow.set_tracking_uri("http://127.0.0.1:5000")

app = Flask(__name__)


@app.route("/recommend/<user_id>")
def recommend_user(user_id):

    data = get_data()

    unrated_books = data[(data["user_id"] != user_id)]["isbn"].unique()

    model_name = "recommender"

    model_version = 2

    model_uri = f"models:/{model_name}/{model_version}"

    loaded_model = mlflow.pyfunc.load_model(model_uri)

    input_data = pd.DataFrame(
        {"user_id": [user_id] * len(unrated_books), "item_id": unrated_books}
    )

    predictions = loaded_model.predict(input_data)

    return (
        pd.DataFrame({"predictions": predictions, "books": unrated_books})
        .sort_values(by="predictions", ascending=False)
        .iloc[:10]["books"]
        .tolist()
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4444)
