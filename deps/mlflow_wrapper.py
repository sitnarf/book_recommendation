import tempfile

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.pyfunc import PythonModel


import mlflow.pyfunc
import pickle
from surprise import Dataset, SVD, Reader


class SurpriseModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, surprise_model):
        self.surprise_model = surprise_model

    def load_context(self, context):
        # Load the Surprise model artifact (if needed during loading)
        with open(context.artifacts["model_artifact"], "rb") as f:
            self.surprise_model = pickle.load(f)

    def predict(self, context, model_input):
        # The model_input is expected to be a DataFrame with user and item columns
        predictions = []
        for _, row in model_input.iterrows():
            user_id = row["user_id"]
            item_id = row["item_id"]
            pred = self.surprise_model.predict(user_id, item_id)
            predictions.append(pred.est)
        return predictions

    def save_model(self):
        # Save the Surprise model to a temp file using pickle
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        with open(temp_file.name, "wb") as f:
            pickle.dump(self.surprise_model, f)
        return temp_file.name
