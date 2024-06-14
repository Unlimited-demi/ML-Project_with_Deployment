
import os
import sys
sys.path.append('C:\\Users\\USER\\Desktop\\mlprojects\\src')
from exception import CustomException
from logger import logging
import dill 
from isort import file
import pandas as pd
import numpy as np



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path , exist_ok =True)

        with open(file_path , "wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e , sys) # type: ignore
      