from dataclasses import dataclass
import os 
import sys
from utils import save_object
from isort import file 
sys.path.append('C:\\Users\\USER\\Desktop\\mlprojects\\src')
from exception import CustomException
from logger import logging

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from utils import evaluate_models

@dataclass

class ModelTrainerConfig():
    trained_model_file_path = os.path.join('artifacts' , "model.pkl")

class Model_Trainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    
    def initiate_model_trainer(self , train_array , test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train , Y_train , X_test , Y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "RandomForest": RandomForestRegressor(),
                "DecisionTree" : DecisionTreeRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "LinearRegression": LinearRegression(),
                "K-Neigbors Regressor": KNeighborsRegressor(),
                "Adaboost" : AdaBoostRegressor(),
                "Catboost Regressor" : CatBoostRegressor(),
                "XGBoost Regressor" : XGBRegressor(),

            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "RandomForest":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            models_report:dict = evaluate_models(X_train=X_train , Y_train =Y_train, X_test = X_test, Y_test = Y_test , models = models,para = params) # type: ignore

            best_model_score  =  max(sorted(models_report.values()))
            best_model_name  = list(models_report.keys())[list(models_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best model found') # type: ignore
            logging.info("Best found model on both training and test_data")

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(Y_test , predicted)

            return r2_square
        except Exception as e :
           raise CustomException(e,sys) # type: ignore

