import logging
import mlflow
import pandas as pd
from typing import Union
from model.model_dev import (
    HyperparameterTuner,
    LightGBMModel,
    LinearRegressionModel,
    RandomForestModel,
    XGBoostModel,
)
from sklearn.base import RegressorMixin
from zenml import step
from zenml.client import Client

from steps.config import ModelNameConfig

experiment_tracker = Client().active_stack.experiment_tracker


@step
def train_model(
    x_train: Union[pd.DataFrame, None],
    x_test: Union[pd.DataFrame, None],
    y_train: Union[pd.Series, None],
    y_test: Union[pd.Series, None],
    config: ModelNameConfig,
) -> RegressorMixin:
    """
    Args:
        x_train: pd.DataFrame or None
        x_test: pd.DataFrame or None
        y_train: pd.Series or None
        y_test: pd.Series or None
    Returns:
        model: RegressorMixin
    """
    try:
        model = None
        tuner = None

        if config.model_name == "lightgbm":
            mlflow.lightgbm.autolog()
            model = LightGBMModel()
        elif config.model_name == "randomforest":
            mlflow.sklearn.autolog()
            model = RandomForestModel()
        elif config.model_name == "xgboost":
            mlflow.xgboost.autolog()
            model = XGBoostModel()
        elif config.model_name == "linear_regression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
        else:
            raise ValueError("Model name not supported")

        tuner = HyperparameterTuner(model, x_train, y_train, x_test, y_test)

        if config.fine_tuning:
            best_params = tuner.optimize()
            trained_model = model.train(x_train, y_train, **best_params)
        else:
            # Ensure that the dropped columns are also dropped from x_test
            if x_test is not None:
                x_test = x_test.drop(
                    [   
                        "title_year",
                    ],
                    axis=1,
                )
            trained_model = model.train(x_train, y_train)
        return trained_model
    except Exception as e:
        logging.error(e)
        raise e
