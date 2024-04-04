# Machine Learning Projects Overview

This repository contains two machine learning projects focusing on job standardization and animal shelter outcomes prediction. Each project employs a variety of data preprocessing, feature engineering, and modeling techniques using Python's scientific and data analysis libraries.

## Project 1: Job Title Standardization

### Objective

To standardize job titles by mapping them to the Standard Occupational Classification (SOC) system using NLP techniques and machine learning models.

### Dependencies

- pandas
- numpy
- langdetect
- matplotlib
- seaborn
- re
- wordcloud
- nltk
- sklearn

### How to Run

1. Ensure you have Python installed on your system.
2. Install the required dependencies using `pip install pandas numpy langdetect matplotlib seaborn nltk sklearn wordcloud`.
3. Download the `jobs.csv` and `soc2020volume1structureanddescriptionofunitgroupsexcel180523.xlsx` files and place them in the project directory.
4. Run the script with `python job_title_standardization.py`.

### Key Features

- Language detection for job descriptions.
- Text preprocessing including tokenization, lemmatization, and removal of stop words.
- TF-IDF vectorization for text data.
- Cosine similarity to map job titles to SOC titles.
- Random Forest Classifier to predict the SOC title for new job descriptions.

## Project 2: Animal Shelter Outcomes Prediction

### Objective

To predict the outcomes of animals in a shelter using their intake conditions, breeds, and other factors.

### Dependencies

- pandas
- sklearn
- matplotlib
- seaborn
- joblib

### How to Run

1. Ensure Python is installed on your system.
2. Install the required dependencies using `pip install pandas sklearn matplotlib seaborn joblib`.
3. Download the `aac_intakes_outcomes.csv` dataset and place it in the project directory.
4. Run the script with `python animal_shelter_outcomes.py`.

### Key Features

- Handling of missing values and feature engineering.
- Encoding of categorical variables and standardization of numerical variables.
- Recursive feature elimination with cross-validated selection of the best number of features.
- Random Forest Classifier for predicting animal outcomes.
- Pipeline integration for preprocessing and modeling steps.

