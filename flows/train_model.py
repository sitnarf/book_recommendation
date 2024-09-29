from pandas import DataFrame
from prefect import task
from surprise import KNNWithZScore

from deps.data import transform_data
from flows.get_data import get_data

from surprise.model_selection import cross_validate


@task
def train_knn_model(data: DataFrame):
    model = KNNWithZScore()
    model.fit(data)
    return model


def train_and_evaluate_knn_model():
    import mlflow

    mlflow.start_run()

    data = get_data()

    model = KNNWithZScore()

    # Perform cross-validation
    cv_results = cross_validate(model, data, cv=5)

    # Log metrics to mlflow
    for metric in cv_results:
        mlflow.log_metric(metric, cv_results[metric].mean())

    mlflow.end_run()

    return cv_results


# Execute the flow
if __name__ == "__main__":
    train_and_evaluate_knn_model()
