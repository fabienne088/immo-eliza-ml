# So we can save the encoding object we used in the X-train into a pickle file, as well as save the scaling object into a pickle file.
# And when we are going to make a new prediction with new data:
# We load the encoding prickle file, use it to transform the new data, then do the same with the scaler, and then with the model.

import pickle

# Load the object from the file
with open('preprocess.pkl', 'rb') as file:
    loaded_preprocess = pickle.load(file)

# Load the model from the file
with open('random_forest_regressor.pkl', 'rb') as file:
    loaded_regressor = pickle.load(file)