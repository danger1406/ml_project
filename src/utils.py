from src.execption import CustomException
import os
import sys
#we will store all the common functionlities so that we can reuse
import numpy as np 
import pandas as pd
from sklearn.metrics import r2_score
import dill
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(x_train,y_train,x_test,y_test,models,param):
    try:
        report={}
        for i in range(len(list(models.values()))):
            model=list(models.values())[i]
            para=param[list(models.keys())[i]]
            gs =GridSearchCV(model,para,cv=3)
            gs.fit(x_train,y_train)
            
            # After doing a grid search CV, you will get the optimal parameters. We will use those parameters for the model so that they give the most accurate result.
            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)
            
            y_train_pred=model.predict(x_train)
            y_test_pred=model.predict(x_test)
            
            train_model_score =r2_score(y_train,y_train_pred)
            test_model_score =r2_score(y_test,y_test_pred)
            
            report[list(models.keys())[i]]=test_model_score
        
        return report
    except Exception as e:
        raise CustomException(e,sys)
        
        