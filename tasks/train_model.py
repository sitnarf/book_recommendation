from prefect import task
import pickle
from sklearn.ensemble import RandomForestClassifier

@task
def train_model(input_data_path: str, model_save_path: str):
    """
    Train a model and save the trained model as a file.
    """
    # Load preprocessed data
    data = pd.read_csv(input_data_path)
    X = data.drop("target", axis=1)
    y = data["target"]

    # Train a model
    model = RandomForestClassifier()
    model.fit(X, y)

    # Save the model to file
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
    return model_save_path
