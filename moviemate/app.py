# load import statements
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(file_path, separation_type=None):
    """
    Load the dataset using pandas and inspect its structure.
    This load is for the purpose of returning a pandas.DataFrame which will 
    later get partitioned into test and train data.
    :param file_path: path to data file that will be loaded
    :param separation_type: way that the file is separated, tab (`\t`) or bar (`|`)
    :return: pandas.Dataframe of the dataset
    """
    # Determine the correct separator
    if separation_type not in ['\t', '|', ',']:
        raise ValueError("Unsupported separation type.")

    # Load the dataset
    df = pd.read_csv(file_path, sep=separation_type, header=None)
    return df


def partition_data(ratings_df, split: float = .8, partition_type='stratified', stratify_by='user_id'):
    """
    Split the data into training and testing sets using user-stratified sampling and temporal sampling.
    :param partition_type: partitioning strategy. stratified (Stratified Sampling), or temporal (Temporal Sampling).
    :return: A tuple containing:
        - train_df: pandas.DataFrame 
            training dataset
        - test_df: pandas.DataFrame 
            testing dataset
    """
    if partition_type == 'stratified':
        if stratify_by not in ratings_df.columns:
            raise ValueError(f"Stratify column '{stratify_by}' not in DataFrame")
        
        # Stratified sampling by stratify_by column
        unique_stratify = ratings_df[stratify_by].unique()
        if len(unique_stratify) < 2:
            raise ValueError("Stratification requires at least two unique values in the stratify_by column.")
        

        unique_train, unique_test = train_test_split(
            unique_stratify,
            train_size=split,
            random_state=42
        )

        # Partition the original dataframe based on stratify_by column
        train_df = ratings_df[ratings_df[stratify_by].isin(unique_train)].copy()
        test_df = ratings_df[ratings_df[stratify_by].isin(unique_test)].copy()
    
    elif partition_type == 'temporal':
        if 'timestamp' not in ratings_df.columns:
            raise ValueError("Temporal partitioning requires a timestamp column.")
        # Sort by timestamp and split
        ratings_df = ratings_df.sort_values('timestamp')
        cutoff = int(len(ratings_df) * split)
        train_df = ratings_df.iloc[:cutoff]
        test_df = ratings_df.iloc[cutoff:]

    else:
        raise ValueError("Unsupported partition_type. Choose 'stratified' or 'temporal'.")

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
