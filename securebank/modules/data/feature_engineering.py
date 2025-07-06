import pandas as pd
import numpy as np
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from modules.data.data_transformer import (
    UppercaseTransformer,
    DropColumnsTransformer,
    ComputeAgeTransformer,
    ConvertDatesTransformer,
    ComputeAverageTransformer,
    FrequencyEncodingTransformer,
    CyclicalEncodingTransformer,
)
import sklearn


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Column groups 
        self.drop_cols = ['first', 'last', 'unix_time', 'cust_long', 'cust_lat', 'street', 'city', 'zip', 'trans_num']
        self.freq_cols = ['cc_num', 'merchant', 'state', 'job']
        self.cyclical_cols = ['day_of_week', 'hour', 'minute', 'seconds', 'day_date', 'month_date']
        self.standardize_cols = [
            'amt', 'merch_lat', 'merch_long', 'city_pop', 'age', 'year_date',
            'cc_num_freq', 'merchant_freq', 'state_freq', 'job_freq',
            'avg_amt_by_cc_num', 'avg_amt_by_merchant', 'avg_amt_by_state',
        ]
        self.onehot_cols = ['sex', 'category']

        # Inner transformers
        self.pipeline = None
        self.column_names_ = None

    def fit(self, X, y=None):
        X = X.copy()
        # Step 1: Apply custom logic before ColumnTransformers
        pre_pipe = Pipeline(steps=[
            ("uppercase", UppercaseTransformer()),
            ("drop_cols", DropColumnsTransformer(columns_to_drop=self.drop_cols)),
            ("compute_age", ComputeAgeTransformer()),
            ("convert_dates", ConvertDatesTransformer()),
            ("compute_avg", ComputeAverageTransformer(columns_to_group=['cc_num', 'merchant', 'state'], value_column='amt')),
            ("frequency_encode", FrequencyEncodingTransformer(columns=self.freq_cols)),
            ("cyclical_encode", CyclicalEncodingTransformer(columns=self.cyclical_cols)),
        ])

        # X_pre = pre_pipe.fit_transform(X) #TODO

        # Step 2: Define ColumnTransformer
        if sklearn.__version__ >= "1.2":
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        else:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

        self.col_transformer = ColumnTransformer(transformers=[
            ('onehot', ohe, self.onehot_cols),
            ('scale', StandardScaler(), self.standardize_cols)
        ], remainder='passthrough')

        # Step 3: Fit on transformed data
        # self.col_transformer.fit(X_pre) #TODO
        # Store the full pipeline for transform()
        self.pipeline = Pipeline(steps=[
            ('pre_custom', pre_pipe),
            ('post_transform', self.col_transformer)
        ])
        self.pipeline.fit(X)

        # Step 4: Save final column names
        X_pre = pre_pipe.fit_transform(X)
        onehot_features = self.col_transformer.named_transformers_['onehot'].get_feature_names_out(self.onehot_cols)
        passthrough_cols = [col for col in X_pre.columns if col not in self.onehot_cols + self.standardize_cols]
        self.column_names_ = list(onehot_features) + self.standardize_cols + passthrough_cols

        return self

    def transform(self, X):
        X_t = self.pipeline.transform(X)
        return pd.DataFrame(X_t, columns=self.column_names_, index=X.index)

    def get_feature_names_out(self):
        return self.column_names_
    
    def save_pipeline(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

