import pickle
from statistics import mean, stdev

import mlflow
import ray
from jsonschema.exceptions import best_match
from numpy import arange
from pandas import DataFrame
from prefect import task, flow
from sklearn.model_selection import KFold
from surprise import KNNWithZScore, SVD
from surprise.model_selection import (
    RandomizedSearchCV,
    KFold as SurpriseKFold,
    cross_validate,
)

from deps.data import transform_data_for_surprise
from deps.mlflow_wrapper import SurpriseModelWrapper
from flows.get_data import get_data


def get_cv():
    return SurpriseKFold(n_splits=10)


def perform_tuned_cv(Model, data, param_grid):
    data_processed = transform_data_for_surprise(data)
    grid_search = RandomizedSearchCV(
        Model, param_grid, measures=["mae", "rmse"], cv=get_cv(), refit=True, n_jobs=-1
    )
    grid_search.fit(data_processed)
    return grid_search, grid_search.best_score, grid_search.best_params["mae"]


def perform_cv(Model, data):
    data_processed = transform_data_for_surprise(data)
    cv_results_nmf = cross_validate(Model(), data_processed, cv=10, n_jobs=-1)
    mean_metrics = {
        "mae": mean(cv_results_nmf["test_mae"]),
        "rmse": mean(cv_results_nmf["test_rmse"]),
    }
    return mean_metrics, {}


@flow(name="Train and evaluate models", log_prints=True)
def train_and_evaluate_svd_model():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    data = get_data()

    param_grid = {
        "n_epochs": list(range(10, 50)),
        "lr_all": list(arange(0.005, 0.10, 0.001)),
        "reg_all": list(arange(0.02, 0.05, 0.01)),
    }

    with mlflow.start_run(run_name="svd optimized"):
        model, best_score, best_params = perform_tuned_cv(SVD, data, param_grid)
        mlflow.log_metric("mae_score", best_score["mae"])
        mlflow.log_metric("rmse_score", best_score["rmse"])
        mlflow.log_params(best_params)

        model_wrapped = SurpriseModelWrapper(model)

        mlflow.pyfunc.log_model(
            artifact_path="surprise_model",
            python_model=model_wrapped,
            artifacts={"model_artifact": model_wrapped.save_model()},
        )

    with mlflow.start_run(run_name="svd default"):
        best_score, best_params = perform_cv(SVD, data)
        mlflow.log_metric("mae_score", best_score["mae"])
        mlflow.log_metric("rmse_score", best_score["rmse"])
        mlflow.log_params(best_params)


# Execute the flow
if __name__ == "__main__":
    train_and_evaluate_svd_model()
