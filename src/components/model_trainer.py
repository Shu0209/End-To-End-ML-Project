import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    train_model_file_path=os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and testing input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "Linear Regression":LinearRegression(),
                "Decision Tree":DecisionTreeRegressor(),
                "K NearestNeighbors":KNeighborsRegressor(),
                "CatBoost Regressor":CatBoostRegressor(verbose=False),
                "XGBoost":XGBRegressor(),
                "Random Forest":RandomForestRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "AdaBoost":AdaBoostRegressor()
            }

            params={
                "Linear Regression":{   
                },
                "Decision Tree":{
                    "criterion": ["squared_error", "friedman_mse"],
                    "splitter": ["best", "random"],
                    "max_depth": [None, 5, 10, 15, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "K NearestNeighbors":{
                    "n_neighbors": [3, 5, 7, 9, 11],
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan", "minkowski"]
                },
                "CatBoost Regressor":{
                    "iterations": [100, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "depth": [4, 6, 8],
                    "l2_leaf_reg": [1, 3, 5, 7]
                },
                "XGBoost":{
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.7, 0.8, 1.0],
                    "colsample_bytree": [0.7, 0.8, 1.0]
                },
                "Random Forest":{
                    "n_estimators": [100, 200],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2"]
                },
                "Gradient Boosting":{
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "AdaBoost":{
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.05, 0.1, 1],
                    "loss": ["linear", "square", "exponential"]
                }


            }

            logging.info("Starting model evaluation with hyperparameter tuning")




            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,param=params)


            best_model_score=max(model_report.values())


            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No Best model found",sys)
            
            logging.info(f"Best model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)

            r2_square=r2_score(y_test,predicted)

            return r2_square


        except Exception as e:
            raise CustomException(e,sys)
        





