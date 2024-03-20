# Model card

## Project context

In the Immo-Eliza-ML project we will build a performant machine learning model to predict prices of real estate proporties in Belgium. This involves cleaning the dataset, preprocessing, model training, model evaluation and iteration.

## Data

### The dataset
A dataset is provided in `data/properties.csv`. Some notes:
- There are about 76 000 properties, roughly equally spread across houses and apartments
- Each property has a unique identifier `id`
- The target variable is `price`
- Variables prefixed with `fl_` are dummy variables (1/0)
- Variables suffixed with `_sqm` indicate the measurement is in square meters
- All missing categories for the categorical variables are encoded as `MISSING` 

### Target variable
The target variable is 'price'.

### Features
Following featueres were used:

['region', 'province', 'total_area_sqm', 'surface_land_sqm', 'nbr_frontages', 'nbr_bedrooms', 'equipped_kitchen', 'fl_furnished', 'fl_open_fire', 'fl_terrace', 'terrace_sqm', 'fl_garden', 'garden_sqm', 'fl_swimming_pool', 'fl_floodzone', 'state_building', 'primary_energy_consumption_sqm', 'epc', 'heating_type', 'fl_double_glazing']


## Model details

Following models were tested:
- Single Linear Regression
- Multiple Linear Regression
- Random Forest Regression

Multiple Linear Regression and Random Forest Regression were finally chosen.

## Performance

Here you can find performance metrics for the various models tested.

**Linear Regression**

Training score: 7.86 %

**Multiple Linear Regression**

| Metric         | Training       | Testing        |
|----------------|----------------|----------------|
| Score          | 32.57%         | 41.22%         |
| MAE            | 171200.16      | 167932.81      |
| MSE            | 166009171623.71| 150780709179.69|
| RMSE           | 407442.23      | 388304.92      |


**Random Forest Regression**

| Metric         | Training       | Testing        |
|----------------|----------------|----------------|
| Score          | 94.06%         | 65.11%         |
| MAE            | 42857.23       | 113583.66      |
| MSE            | 14620510605.64 | 89505049990.32 |
| RMSE           | 120915.30      | 299173.95      |

## Limitations

What are the limitations of your model?

## Usage

What are the dependencies, what scripts are there to train the model, how to generate predictions, ...

## Maintainers

In case of questions or issues you can contact the developer on LinkedIn.<br />
[www.linkedin.com/in/fabienne-thüer](www.linkedin.com/in/fabienne-thüer-56a8a0a)
