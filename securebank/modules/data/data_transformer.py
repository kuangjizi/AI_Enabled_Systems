from sklearn.base import BaseEstimator, TransformerMixin
from modules.data.raw_data_handler import (
    compute_age,
    convert_dates,
    compute_average_columns,
    cyclical_encode_columns
)

# Convert the custom transformation into standard sklearn preprocessing
class UppercaseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.cat_cols = X.select_dtypes(include='object').columns
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col in self.cat_cols:
            if col in X.columns:
                X_copy[col] = X_copy[col].str.upper()
        return X_copy

class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=[]):
        self.columns_to_drop = columns_to_drop
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        drop_cols = [col for col in self.columns_to_drop if col in X_copy.columns] # drop existing columns
        return X_copy.drop(columns=drop_cols)

class ComputeAgeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return compute_age(X.copy(), columms_to_compute=['trans_date_trans_time', 'dob'])

class ConvertDatesTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return convert_dates(X.copy(), column_to_convert='trans_date_trans_time')

class ComputeAverageTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_group, value_column):
        self.columns_to_group = columns_to_group
        self.value_column = value_column

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return compute_average_columns(X.copy(), columns_to_group=self.columns_to_group, value_column=self.value_column)

class FrequencyEncodingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.freq_maps = {}
        for col in self.columns:
            self.freq_maps[col] = X[col].value_counts(normalize=True).to_dict()
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            X_copy[f'{col}_freq'] = X_copy[col].map(self.freq_maps[col]).fillna(0)
            X_copy = X_copy.drop(col, axis=1) # drop the original column
        return X_copy

class CyclicalEncodingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return cyclical_encode_columns(X.copy(), columns_to_encode=self.columns)
