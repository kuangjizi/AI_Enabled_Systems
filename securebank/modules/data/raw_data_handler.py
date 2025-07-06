import os
import pandas as pd
import numpy as np
from dateutil import parser


# Helper functions for extraction, transformation, and description of raw data for machine learning preprocessing.

def extract(customer_file_path, transaction_file_path=None, fraud_file_path=None):
    """
    Reads raw data files and returns them as DataFrames.
    Input:
        customer_file_path: path to CSV file containing customer information.
        transaction_file_path: path to Parquet file containing transaction data.
        fraud_file_path: path to JSON file containing fraud labels.
    Output: Three pandas DataFrames:
        customer_information
        transaction_information
        fraud_information
    """
    customer_information = pd.read_csv(customer_file_path)

    if transaction_file_path is not None:
        transaction_information = pd.read_parquet(transaction_file_path).reset_index()
    else:
        transaction_information = None

    if fraud_file_path is not None:
        fraud_information = pd.read_json(fraud_file_path, typ='series').to_frame(name='is_fraud').reset_index(names=['trans_num'])
    else:
        fraud_information = None

    return customer_information, transaction_information, fraud_information

def merge_and_clean(customer_information, transaction_information, fraud_information=None):
    """
    Prepares and cleans data by:
        merging the three data sources,
        imputing/dropping missing values,
        and dropping duplicate rows.
    Input:
        customer_information: DataFrame of customer information.
        transaction_information: DataFrame of transaction information.
        fraud_information: DataFrame of fraud information. If None, no fraud information will be merged.
    Output: 
        A cleaned and merged pandas DataFrame. 
        The DataFrame should have all the original columns. 
        Ensure that each row in the resulting DataFrame represents a unique transaction, with all relevant information merged correctly. 
        No additional preprocessing (e.g., scaling or encoding) is required.
    """
    def try_parse_date(s):
        try:
            return parser.parse(s, dayfirst=True)  # try parsing with dayfirst
        except Exception:
            return pd.NaT

    # Drop the "index" columns from transaction and customer data, as not required by the model according to the "Case Study Background" section.
    if "index" in customer_information.columns:
        customer_information.drop('index', axis=1, inplace=True) 
    if "index" in transaction_information.columns:
        transaction_information.drop('index', axis=1, inplace=True)

    # Rename columns to ensure consistency
    customer_information.rename(columns={"lat": "cust_lat", "long": "cust_long"}, inplace=True)

    # Parsing timestemp columns
    customer_information['dob'] = customer_information['dob'].apply(try_parse_date) # parsting the dob column
    transaction_information['trans_date_trans_time'] = pd.to_datetime(transaction_information['trans_date_trans_time'])

    # Merge the three data sources
    if fraud_information is not None:
        merged_data = transaction_information.merge(customer_information, on='cc_num', how='left', suffixes=('', '_drop')).merge(fraud_information, on='trans_num', how='left', suffixes=('', '_drop'))
        assert merged_data.shape[0] == transaction_information.shape[0], "The number of rows in the merged data does not match the transaction data." # Ensure all transactions are present after merging
        assert merged_data.shape[1] == customer_information.shape[1] + transaction_information.shape[1] + fraud_information.shape[1] - 2, "The number of columns in the merged data does not match the sum of the individual data sources." # Ensure all columns are present after merging
    else:
        merged_data = transaction_information.merge(customer_information, on='cc_num', how='left', suffixes=('', '_drop'))
        assert merged_data.shape[0] == transaction_information.shape[0], "The number of rows in the merged data does not match the transaction data." # Ensure all transactions are present after merging
        assert merged_data.shape[1] == customer_information.shape[1] + transaction_information.shape[1] - 1, "The number of columns in the merged data does not match the sum of the individual data sources." # Ensure all columns are present after merging

    merged_data = merged_data.loc[:, ~merged_data.columns.str.endswith('_drop')] # Drop duplicate columns from the merge
    
    # Drop rows with missing values
    merged_data = merged_data.dropna()

    # Drop duplicate rows
    merged_data = merged_data.drop_duplicates()

    return merged_data

def describe(df):
    """
    Produces a dictionary of quality metrics for the transformed data.
    Input:
        df: Input DataFrame.
    Output: A dictionary summarizing:
        Number of records
        Number of columns
        Feature names
        Missing values
        Data types
    """
    description = {}
    description['num_records'] = df.shape[0]
    description['num_columns'] = df.shape[1]
    description['feature_names'] = df.columns.tolist()
    description['missing_values'] = df.isnull().sum().sum() 
    description['data_types'] = [f"{col}: {dtype}" for col, dtype in df.dtypes.items()] # list of str with column name and data type
    
    return description


def compute_age(df, columms_to_compute=['trans_date_trans_time', 'dob']):
    """
    Computes the age of customers based on their date of birth and transaction date.
    Input:
        df: DataFrame with 'dob' column containing date of birth in 'YYYY-MM-DD' format.
        columms_to_compute: the start and end date columns to compute age
    Output: 
        Modified DataFrame with an 'age' column.
    """
    assert len(columms_to_compute) == 2, "The start date and end date columns are needed."

    start_date_col, end_date_col = columms_to_compute[0], columms_to_compute[1]
    df['age'] = (pd.to_datetime(df[start_date_col]) - df[end_date_col]).dt.days // 365  # Calculate age in years
    df.drop(end_date_col, axis=1, inplace=True)  # Drop the dob column after calculating age

    return df

def convert_dates(df, column_to_convert='trans_date_trans_time'):
    """
    Replaces specified timestamp column into seven distinct columnss.
    Input:
        df: DataFrame with trans_date_trans_time column.
        column_to_convert: datetime column to convert
    Output: 
        Modified DataFrame containing the seven new columns (without trans_date_trans_time column). 
        This DataFrame should retain the original column order with new columns append to the end.
    """
    datetime = pd.to_datetime(df.pop(column_to_convert))
    df['day_of_week'] = datetime.dt.day_name()
    df['hour'] = datetime.dt.hour
    df['minute'] = datetime.dt.minute
    df['seconds'] = datetime.dt.second
    df['day_date'] = datetime.dt.day
    df['month_date'] = datetime.dt.month_name()
    df['year_date'] = datetime.dt.year
    
    return df

def compute_average_columns(df, columns_to_group, value_column='amt'):
    """
    Aggregate columns from the DataFrame.
    Input:
        df: DataFrame from which to compute average columns.
        columns_to_group: List of column names to group by.
        value_column: Column to be aggregated
    Output:
        DataFrame with aggregated columns.
    """
    if len(columns_to_group) == 0:
        return df
    
    for col in columns_to_group:
        if col in df.columns:
            new_col = f"avg_{value_column}_by_{col}" # e.g. avg_amt_by_merchant
            df[new_col] = df.groupby(col)[value_column].transform('mean')

    return df

def drop_columns(df, columns_to_drop):
    """
    Drops specified columns from the DataFrame.
    Input:
        df: DataFrame from which to drop columns.
        columns_to_drop: List of column names to drop.
    Output:
        DataFrame with specified columns dropped.
            
    """
    if len(columns_to_drop) == 0:
        return df
    else:
        df = df.drop(columns=columns_to_drop, errors='ignore')
        return df

def standardize_columns(df, columns_to_standardize):
    """
    Standardizes the specified columns in the DataFrame.
    Input:
        df: DataFrame containing the data to standardize.
        columns_to_standardize: List of column names to standardize.
    Output:
        DataFrame with specified columns standardized.
    """
    if len(columns_to_standardize) == 0:
        return df
    else:
        for col in columns_to_standardize:
            if col in df.columns:
                df[col] = (df[col] - df[col].mean()) / df[col].std()
        return df

def onehot_encode_columns(df, columns_to_encode):
    """
    Encodes specified categorical columns in the DataFrame using one-hot encoding.
    Input:
        df: DataFrame containing the data to encode.
        columns_to_encode: List of column names to encode.
    Output:
        DataFrame with specified columns encoded.
    """
    if len(columns_to_encode) == 0:
        return df
    else:
        df = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)
        return df

def cyclical_encode_columns(df, columns_to_encode):
    """
    Encodes specified cyclical columns in the DataFrame using sine and cosine transformations.
    Input:
        df: DataFrame containing the data to encode.
        columns_to_encode: List of column names to encode.
    Output:
        DataFrame with specified columns encoded.
    """
    if len(columns_to_encode) == 0:
        return df
    else:
        for col in columns_to_encode:
            if col in df.columns:
                if col == 'day_of_week': 
                    df['day_of_week'] = df['day_of_week'].map({
                        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                        'Friday': 4, 'Saturday': 5, 'Sunday': 6
                    })
                
                if col == 'month_date':
                    df['month_date'] = df['month_date'].map({
                        'January': 1, 'February': 2, 'March': 3, 'April': 4,
                        'May': 5, 'June': 6, 'July': 7, 'August': 8,
                        'September': 9, 'October': 10, 'November': 11, 'December': 12
                    })

                df[col + '_sin'] = np.sin(2 * np.pi * df[col] / df[col].max())
                df[col + '_cos'] = np.cos(2 * np.pi * df[col] / df[col].max())
                df.drop(col, axis=1, inplace=True)

        return df

def frequency_encode_columns(df, columns_to_encode):
    """
    Encodes specified categorical columns in the DataFrame using frequency encoding.
    Input:
        df: DataFrame containing the data to encode.
        columns_to_encode: List of column names to encode.
    Output:
        DataFrame with specified columns encoded.
    """
    if len(columns_to_encode) == 0:
        return df
    else:
        for col in columns_to_encode:
            if col in df.columns:
                freq = df[col].value_counts()
                df[col + '_freq'] = df[col].map(freq)
                df.drop(col, axis=1, inplace=True)
        return df

def check_outliers(df, columns_to_check):
    """
    Checks for outliers in specified columns of the DataFrame using the IQR method.
    Input:
        df: DataFrame containing the data to check.
        columns_to_check: List of column names to check for outliers.
    Output:
        A dictionary with column names as keys and a list of outlier indices as values.
    """
    if len(columns_to_check) == 0:
        raise ValueError("Please specify columns for outlier checks")
    
    outliers = {}
    for col in columns_to_check:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
            outliers[col] = outlier_indices
    return outliers 


def check_data_quality(df):
    """
    Checks the quality of the data.
    Input:
        df: DataFrame containing the data to check.
    Output:
        A dictionary containing data quality metrics:
            - dup_rows: Number of duplicate rows in the DataFrame.
            - null_check: DataFrame with columns, null value counts, percentage of total, and data types.
            - cols_with_nulls: List of columns that contain null values.
            - unique_values: Series with the count of unique values for each object type column.
            - cols_with_constants: List of columns that have constant values (i.e., only one unique value).
    """
    results = {}
    # Dup Check
    results["dup_rows"] = df.duplicated().sum()

    # Null Check
    null_check = df.isnull().sum().to_frame(name='Null Value Count')
    null_check['Pct of Total'] = null_check['Null Value Count'] * 100.0 / df.shape[0]
    null_check['Pct of Total'] = null_check['Pct of Total'].apply(lambda x: float("{:.2f}".format(x)))
    dtypes_df = pd.DataFrame(df.dtypes, columns=['DataType'])
    null_check = pd.concat([null_check, dtypes_df], axis=1).reindex(null_check.index)
    cols_with_nulls = null_check[null_check['Null Value Count']>0].index.tolist()

    results["null_check"] = null_check
    results["cols_with_nulls"] = cols_with_nulls

    # Unique value count
    results["unique_values"] = df.select_dtypes(include='object').nunique()

    # Constant columns         
    cols_with_constants = df.columns[df.nunique() <= 1]
    cols_with_constants = [c for c in cols_with_constants if c != 'Label'] 
    results["cols_with_constants"] = cols_with_constants

    return results

def check_class_distribution(df, target_column='is_fraud'):
    """
    Computes the class distribution of the target column in the input DataFrame.
    Input:
        df: DataFrame containing the data to analyze.
        target_column: The column for which to compute the class distribution.
    Output:
        A dictionary with the class distribution.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in input data.")
    
    class_counts = df[target_column].value_counts().to_dict()
    return class_counts

     



# Test the handler
# if __name__ == '__main__':
#     storage_path, save_path = 'securebank/data_sources/', 'securebank/data_output/'
#     handler = RawDataHandler(storage_path, save_path)
#     customer_info, transaction_info, fraud_info = handler.extract('customer_release.csv', 'transactions_release.parquet', 'fraud_release.json')
#     # print(handler.describe(customer_info))
#     # print(handler.describe(transaction_info))
#     # print(handler.describe(fraud_info))

#     # Transform the data
#     cleaned_data = handler.transform(customer_info, transaction_info, fraud_info)
#     cleaned_data = handler.convert_dates(cleaned_data)
#     print(cleaned_data.head(10)) 

#     # Describe the cleaned data
#     description = handler.describe(cleaned_data)
#     print(description)