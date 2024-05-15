from numpy import str_
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml import pipeline
from zenml.steps.base_step import BaseStep
from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.ingest_data import ingest_df
from steps.model_train import train_model

docker_settings = DockerSettings(required_integrations=[MLFLOW])


@pipeline(enable_cache=True, settings={"docker": docker_settings})
def train_pipeline(data_path: str):
    """
    Args:
        ingest_data: DataClass
        clean_data: DataClass
        model_train: DataClass
        evaluation: DataClass
    Returns:
        mse: float
        rmse: float
    """
    df = ingest_df(data_path)
    x_train, x_test, y_train, y_test = clean_data(df)
    model = train_model(x_train, x_test, y_train, y_test)
    mse, rmse = evaluation(model, x_test, y_test)
 