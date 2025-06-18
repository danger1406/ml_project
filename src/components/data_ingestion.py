import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    #The artifacts folder is used to store the train outputs. 
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")
    raw_data_path:str=os.path.join("artifacts","raw.csv")
    # These are inputs that are given to the data ingestion component 
    # So now this number occurs. Where to save the files? It may be train or test. 
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        # So when we create this class, the three paths will be saved in the class variable. 
        
        
        
    def initiate_data_ingestion(self):
        # If the data is present in other sources, we can write the code to retrieve the data from those sources here. 
        logging.info("Entered the data ingestion method or componenet")
        try:
            df=pd.read_csv(r"notebook\data\stud.csv")
            logging.info('Read the dataset as dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
    #   You are getting the directory name of that specific path and creating that if it does not exist  
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            #convert to csv file the raw data paths 
            logging.info("Train test split is initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of the data is completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
            
if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()
    
        