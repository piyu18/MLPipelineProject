import os, sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformationConfig:
    preprocess_file_path = os.path.join('artifacts/data_transformation', 'preprocess.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformation_obj(self):
        try:
            logging.info('Data Transformation started')
            numerical_features = ['age', 'workclass', 'education', 'marital_status', 'occupation',
       'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
       'hours_per_week','native_country']
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median',)),
                    ('scaler', StandardScaler())
                    ]
            )
            preprocessor = ColumnTransformer([
                'num_pipelines', num_pipeline, numerical_features
            ])
            return preprocessor
        except Exception as e:
            raise CustomException
        
    def remove_outliers_IQR(self, col, df):
        try:
            logging.info('Removing outliers')
            Q1 = df[col].quartile(0.25)
            Q3 = df[col].quartile(0.75)
            iqr = Q3 - Q1

            upper_limit = Q3 + 1.5 * iqr
            lower_limit = Q1 - 1.5 * iqr

            df.loc[(df[col]>upper_limit), col] = upper_limit
            df.loc[(df[col]<lower_limit), col] = lower_limit

            return df

        except Exception as e:
            raise CustomException
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            numerical_features = ['age', 'workclass', 'education', 'marital_status', 'occupation',
       'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
       'hours_per_week','native_country']
            
            for col in numerical_features:
                self.remove_outliers_IQR(col=col, df=train_data)
            logging.info('Removed outlier for training data')

            for col in numerical_features:
                self.remove_outliers_IQR(col=col, df=test_data)
            logging.info('Removed outlier for testing data')

            preprocess_obj = self.get_data_transformation_object()
            target_column = "income"
            target = [target_column]
            logging.info('Splitting train data into dependent and independent features')
            X_train = train_data.drop(target, axis=1)
            y_train = train_data[target_column]
            logging.info('Splitting test data into dependent and independent features')
            X_test = test_data.drop(target, axis=1)
            y_test = test_data[target_column]

        except Exception as e:
            raise CustomException