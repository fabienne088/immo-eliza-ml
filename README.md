# IMMO-ELIZA-ML
Preprocess the data and finally build a performant machine learning model.

[![forthebadge made-with-python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

## Description
This notebook shows the analysis of the houses and apartments in the top 30 cities in

The Immo Eliza Data Analysis project focuses on analyzing a dataset of properties including houses and apartments for sale from the website Immoweb. It involves cleaning the dataset, performing exploratory data analysis, and creating visualizations to gain insights into the properties and their characteristics.



![Alt text](src/House_analysis_2.png)


## 📦 Repo structure
   │
├── Notebooks\
│   ├── 1. cleaning.ipynb
│   ├── 2. preprocessing.ipynb
│   ├── 3. model_training_test.ipynb
│   ├── 4. SimpleLinReg.ipynb
│   ├── 5. MultipleLinReg.ipynb
│   ├── 6. RandomForestReg.ipynb
│   └── 7. functions.ipynb
│
├── Scripts\
│   ├── 1. train.ipynb
│   ├── 2. predict.ipynb
│   ├── MultipleLinReg.ipynb
│   ├── RandomForestReg.ipynb
│   └── functions.ipynb
│
├── data\
│   ├── cleaned_properties.csv
│   └── properties.csv
│
├── .gitignore
├── MODELSCARD.md
├── README.md
└── requirements.txt


## ⚙️ Installation

To run the project, you need to install the required libraries. You can click on the badge links to learn more about each library and its specific version used in this project. You can install them manually using pip install <library name> or just running pip install -r requirements.txt.

. Install the required libraries:

   - [![python version](https://img.shields.io/badge/python-3.x-blue)](https://python.org)
   - [![Pandas Version](https://img.shields.io/badge/pandas-2.x-green)](https://pandas.pydata.org/)
   - [![NumPy Version](https://img.shields.io/badge/numpy-1.x-orange)](https://numpy.org/)
   - [![Matplotlib Version](https://img.shields.io/badge/Matplotlib-3.x-red)](https://matplotlib.org/)
   - [![Seaborn Version](https://img.shields.io/badge/seaborn-0.x-yellow)](https://seaborn.pydata.org/)
   - [![sklearn Version](https://img.shields.io/badge/sklearn-0.x-grey)](https://scikit-learn.org/stable/)

   ## The dataset
It includes about 39000 properties scrapped from ImmoWeb: [Immoweb](https://www.immoweb.be) 

To run the program you'll need the requirements.txt containing the required libraries.

1. Clone the repository:
    ```bash
    git clone https://github.com/mahsanazar/immo-eliza-DAMI-analysis.git
    ```

2. Navigate to the project directory:
    ```bash
    cd immo-eliza-DAMI-analysis
    ```

3. You're all set! You can now explore the analysis notebooks in the `analysis` and `reports` directories and work with the data in the `data` directory. Enjoy!

## 🛎️ Usage
To use this repository, follow these steps:

1. **Clone the Repository**: 
    - Clone the repository to your local machine using the following command:
    ```bash
    git clone https://github.com/mahsanazar/immo-eliza-DAMI-analysis.git
    ```

2. **Navigate to the Project Directory**:
    - Once cloned, navigate to the project directory:
    ```bash
    cd immo-eliza-DAMI-analysis
    ```

3. **Explore Analysis Notebooks**:
    - The `analysis` directory contains Jupyter notebooks (`*.ipynb`) where you can explore various analyses performed on the data. Open these notebooks in Jupyter Notebook or JupyterLab to view the analyses and results.

4. **Access Reports**:
    - The `reports` directory includes reports generated from the analysis. These reports may contain visualizations, insights, and conclusions drawn from the data analysis.

5. **Work with Data**:
    - The `data` directory contains the dataset used for analysis. You can find both raw and clean versions of the dataset. Explore the data files to understand their structure and contents.

## 📑 Sources
- [Immoweb](https://www.immoweb.be/en) - Real estate website from which data is scraped.


## ⏱️ Timeline
This project took form in six days.

## 📌 Personal Situation
This project was made during the AI Bootcamp at BeCode.