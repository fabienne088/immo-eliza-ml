# Import libraries
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

def explore_df(df):
    """ This function reads a CSV file, displays basic information about 
    the df, and returns descriptive statistics of the data.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        df (DataFrame): The DataFrame read from the CSV file.
    """
    # Read the CSV file into a DataFrame
    #df = pd.read_csv(file_path)

    # Display the first lines of the df
    print(df.head().T)
    print()

    # Display the number of proporties and features
    num_properties, num_features = df.shape
    print(f"There are {num_properties} proporties and {num_features} features.\n")

    # Display the features:
    print(f"The features are: {', '.join([str(feature)for feature in df.columns])}.\n")
     
    # Display the descriptive statistics
    print(df.describe(include="all").T)
    print()

    # return the df
    return df

# Specify the file path
df = pd.read_csv(r"data\cleaned_properties.csv")
# Call the function to explore the data
explore_df(df)

def filter_houses(df):
    """Filter out the DataFrame for values where the property_type is 'HOUSE'
    and the subproperty_type is not 'APARTMENT_BLOCK'.

    Args:
        df (DataFrame): Input DataFrame. 

    Returns:
        Dataframe: Filtered DataFrame containing only HOUSE properties.
    """    
    # Filter out the DataFrame for values APARTMENT and APARTMENT_BLOCK
    df_house = df[(df["property_type"] == "HOUSE") & (df['subproperty_type'] != 'APARTMENT_BLOCK')]
    # df_house.info()

    return df_house

# Call the filter_houses function and pass your DataFrame as an argument
df_house = filter_houses(df)

def prepare_data(df_house):
    """Prepare the data for machine learning by splitting it into features (X) and target variable (y),
    and then splitting it into training and test sets.

    Parameters:
        df_house (DataFrame): DataFrame containing the subset of houses data.

    Returns:
        X_train (DataFrame): Features for training.
        X_test (DataFrame): Features for testing.
        y_train (Series): Target variable for training.
        y_test (Series): Target variable for testing.
    """    
    # Name X and y
    X = df_house.drop(columns=['price', 'subproperty_type', 'property_type', 'zip_code', 'locality', 'construction_year', 'cadastral_income'])
    y = df_house['price']

    # Split the data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    
    # X_train.info()
    # Display the features:
    print(f"The features of df_house are:\n {', '.join([str(feature)for feature in X_train.columns])}.\n")

    return X_train, X_test, y_train, y_test

# Call prepare_data and pass df_house as an argument:
X_train, X_test, y_train, y_test = prepare_data(df_house)


def preprocess_data(X_train, X_test):
    """Preprocesses training and test data including imputation, encoding, and scaling.

    Parameters:
        X_train (DataFrame): Input training DataFrame.
        X_test (DataFrame): Input test DataFrame.

    Returns:
        tuple of pandas DataFrames: Preprocessed training and test DataFrames.
    """    
    # Separate numerical and categorical columns
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns

    # Define preprocessing steps for numerical and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Fit and transform the preprocessing steps on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Convert the processed data into DataFrames
    X_train_processed = pd.DataFrame(X_train_processed, columns=numeric_cols.tolist() +
                                     preprocessor.named_transformers_['cat']
                                     .named_steps['onehot'].get_feature_names_out(categorical_cols).tolist())
    X_test_processed = pd.DataFrame(X_test_processed, columns=numeric_cols.tolist() +
                                    preprocessor.named_transformers_['cat']
                                    .named_steps['onehot'].get_feature_names_out(categorical_cols).tolist())

    return X_train_processed, X_test_processed

#  Call preprocess_data
X_train_processed, X_test_processed = preprocess_data(X_train, X_test)
