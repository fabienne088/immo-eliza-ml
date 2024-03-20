# Model card

## Project context

In the Immo-Eliza-ML project we will build a performant machine learning model to predict prices of real estate proporties in Belgium. This involves cleaning the dataset, preprocessing, model training, model evaluation and iteration.

## Data

### The dataset
A dataset is provided in [proporties](data\properties.csv). Some notes:
- There are about 76 000 properties, roughly equally spread across houses and apartments
- Each property has a unique identifier `id`
- The target variable is `price`
- Variables prefixed with `fl_` are dummy variables (1/0)
- Variables suffixed with `_sqm` indicate the measurement is in square meters
- All missing categories for the categorical variables are encoded as `MISSING` 

### Target variable
The target variable is 'price'

### Features
Following featueres were tested:
['region', 'province', 'total_area_sqm', 'surface_land_sqm', 'nbr_frontages', 'nbr_bedrooms', 'equipped_kitchen', 'fl_furnished', 'fl_open_fire', 'fl_terrace', 'terrace_sqm', 'fl_garden', 'garden_sqm', 'fl_swimming_pool', 'fl_floodzone', 'state_building', 'primary_energy_consumption_sqm', 'epc', 'heating_type', 'fl_double_glazing']


## Model details

Following models were tested:
- Single Linear Regression
- Multiple Linear Regression
- Random Forest Regression


final model chosen, ...

## Performance

Performance metrics for the various models tested, visualizations, ...

## Limitations

What are the limitations of your model?

## Usage

What are the dependencies, what scripts are there to train the model, how to generate predictions, ...

## Maintainers

In case of questions or issues you can contact the developer on LinkedIn.
https://www.linkedin.com/in/fabienne-th%C3%BCer-56a8a0a?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BeETk08eKRMStMfq9JmDuYA%3D%3D