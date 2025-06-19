#Components are the modules that we are specifically going to use in the project. 

import os
import sys
from dataclasses import dataclass
# For every file we will write a config file so that we can create the paths that are needed
from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostClassifier,GradientBoostingClassifier,RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.execption import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def intitiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting Training and testing data")
            # Xtrain and y train lo combine aii untai dependend and indepen features 
            x_train,x_test,y_train,y_test=(
                train_array[:,:-1],
                test_array[:,:-1],
                train_array[:,-1],
                test_array[:,-1]
            )
            models = {
            "Linear Regression": LinearRegression(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "XGBRegressor": XGBRegressor(), 
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),  # 'verbose' controls the amount of logging output during training. Setting verbose=False disables training logs.
            }
            # Now we will create a function that evaluates the data for each model and gives the ouput 
            
            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
            # The function will be created in utils file
            
            # We will take the best score of all the models that we have tested 
            best_model_score=max(sorted(model_report.values()))
            
            # To ge the name of the model that had got this score using the index method 
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model=models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("Completed training and testing of data and found the model")
            
            save_object(
                file_path=ModelTrainerConfig.trained_model_file_path,
                obj=best_model
            )
            
            predicted=best_model.predict(x_test)
            model_r2_score=r2_score(y_test,predicted)
            return model_r2_score
        except Exception as e:
            raise CustomException(e,sys)
            
        