# So we can save the encoding object we used in the X-train into a pickle file, as well as save the scaling object into a pickle file.
# And when we are going to make a new prediction with new data:
# We load the encoding prickle file, use it to transform the new data, then do the same with the scaler, and then with the model.

# Import libraries
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

# Sample data for the new DataFrame
data = {
    'region': ['Flanders', 'Flanders', 'Brussels-Capital', 'Wallonia'],
    'province': ['East Flanders', 'Antwerp', 'Brussels', 'Namur'],
    'total_area_sqm': [125.5, 185.3, 110.7, 150.2],
    'surface_land_sqm': [680, 180, 505, 710],
    'nbr_frontages': [3, 4, 2, 3],
    'nbr_bedrooms': [3, 4, 2, 3],
    'equipped_kitchen': ['nan', 'HYPER_EQUIPPED', 'INSTALLED', 'USA_UNINSTALLED'],
    'fl_furnished': [1, 0, 1, 0],
    'fl_open_fire': [1, 0, 1, 0],
    'fl_terrace': [1, 1, 0, 1],
    'terrace_sqm': [20, 25, None, 30],
    'fl_garden': [1, 0, 1, 0],
    'garden_sqm': [50, None, 60, None],
    'fl_swimming_pool': [1, 0, 1, 0],
    'fl_floodzone': [0, 1, 0, 1],
    'state_building': ['GOOD', 'AS_NEW', None, 'GOOD'],
    'primary_energy_consumption_sqm': [150, None, 280, None],
    'epc': ['A', 'B', 'C', 'D'],
    'heating_type': ['GAS', 'ELECTRIC', 'FUELOIL', None],
    'fl_double_glazing': [1, 0, 1, 0]
}

# Create a new DataFrame
new_data_df = pd.DataFrame(data)

# Display the new DataFrame
print(new_data_df.head)

import pickle
# Load the preprocessor
with open('preprocessor.pkl', 'rb') as file:
    preprocessor_loaded = pickle.load(file)

# Load the preprocess_data function
with open('preprocessing.pkl', 'rb') as file:
    preprocessing_loaded = pickle.load(file)

# Now you can use preprocess_data_loaded like a regular function
new_data_processed = preprocessing_loaded(new_data_df, preprocessor_loaded)


# Load the model from the file
#with open('random_forest_regressor.pkl', 'rb') as file:
    #loaded_regressor = pickle.load(file)