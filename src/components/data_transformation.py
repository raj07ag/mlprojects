import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.exception import logging
from src.utils import save_obj


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            numerical_features = ['writing_score','reading_score']
            categorical_features = ['gender',
                                    'race_ethnicity',
                                    'parental_level_of_education',
                                    'lunch',
                                    'test_preparation_course']
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical columns: {numerical_features}")
            logging.info(f"Categorical Columns: {categorical_features}")

            preprocessor = ColumnTransformer(
                [
                    ('numerical_pipeline',num_pipeline,numerical_features),
                    ('categorical_pipeline',cat_pipeline,categorical_features)
                ]
            )

            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical Columns encoding completed")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Reading train and test data')

            logging.info("Obtaining preprocessed object")

            preprocessing_obj = self.get_data_transformer_obj()

            target_column = 'math_score'
            numerical_features = ['writing_score', 'reading_score']

            input_features_train_df = train_df.drop(columns=[target_column], axis = 1)
            target_features_train_df = train_df[target_column]

            input_features_test_df = test_df.drop(columns=[target_column], axis = 1)
            target_features_test_df = test_df[target_column]

            logging.info("applying preprocessor object on training and testing dataframe")

            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessing_obj.transform(input_features_test_df)

            train_arr = np.c_[
                input_features_train_arr, np.array(target_features_train_df)
            ]

            test_arr = np.c_[input_features_test_arr, np.array(target_features_test_df)]

            logging.info("saving preprocessing object")

            save_obj(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
            