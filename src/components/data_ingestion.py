import os
import sys
from tarfile import data_filter



sys.path.append('C:\\Users\\USER\\Desktop\\mlprojects\\src')
from exception import CustomException
from logger import logging
from model_trainer import Model_Trainer , ModelTrainerConfig
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from components.data_transformation import DataTransformation
from components.data_transformation import DataTransformationConfig


@dataclass

class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts' , "train_data.csv")
    test_data_path: str = os.path.join('artifacts' , "test_data.csv")
    raw_data_path: str = os.path.join('artifacts' , "raw_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config  = DataIngestionConfig()

    def initiate_data_ingestion(self) -> tuple[str, str, str]:
        logging.info('Entered the data ingestion method or component')
        try:
            df = pd.read_csv('C:\\Users\\USER\\Desktop\\mlprojects\\notebook\\data\\stud.csv')
            logging.info('Data loaded successfully as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path) , exist_ok =True)
      
            df.to_csv(self.ingestion_config.raw_data_path , index=False , header= True)
            logging.info("Train test split initiated")
            train_set , test_set = train_test_split(df,test_size=0.2 , random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path , index=False , header= True)
            test_set.to_csv(self.ingestion_config.test_data_path , index=False , header= True)

            logging.info("Ingestion of data ingestion completed succesfully")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )
         

        except Exception as e:
            raise CustomException(e, sys)  # type: ignore
        
if __name__ == "__main__":
     obj=DataIngestion()
     train_data , test_data , raw_data =obj.initiate_data_ingestion()


     data_transformation = DataTransformation()
     train_arr , test_arr , raw_stuff= data_transformation.initiate_data_transformation(train_data , test_data)

     modeltrainer = Model_Trainer()
     print(modeltrainer.initiate_model_trainer(train_array=train_arr , test_array= test_arr ))
            