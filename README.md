# Brain-Tumor

The project focuses on analyzing tumor data using machine learning techniques, including logistic regression and random forest classification. This README file provides an overview of the project and explains each aspect of the code.

## Table of Contents

- [Introduction](#introduction)
- [Data Overview](#data-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Logistic Regression](#logistic-regression)
- [Random Forest Classification](#random-forest-classification)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction

In this project, we aim to analyze tumor data and predict the presence or absence of a tumor using machine learning algorithms. We employ logistic regression and random forest classification techniques to train models on the provided dataset. The project involves several steps, including data exploration, preprocessing, model training, and evaluation.

## Data Overview
This is a brain tumor feature dataset including five first-order features and eight texture features with the target level (in the column Class).

-	Image: Image name
-	Class: value Tumor = 1 Non tumor =0
-	Mean: First order feature mean
-	Variance: First order feature variance
-	Standard Deviation: First order feature std deviation
-	Entropy: Second order feature entropy
-	Skewness: First order feature skewness
-	Kurtosis: First order feature kurtosis
-	Contrast: Second order feature contrast
-	Energy: Second order feature energy
-	ASM: Second order feature ASM
-	Homogeneity: Second order feature homogeneity
-	Dissimilarity: Second order feature dissimilarity
-	Correlation: Second order feature correlation
-	Coarseness: Second order feature coarness

**Image column defines image name and Class column defines either the image has tumor or not (1 = Tumor, 0 = non-Tumor)**

## Installation

To run the code locally, please ensure you have the following dependencies installed:

- pandas
- seaborn
- matplotlib
- pandasgui
- scikit-learn

You can install these dependencies using pip:

```
pip install pandas seaborn matplotlib pandasgui scikit-learn
```

## Usage

To use the code, follow these steps:

1. Clone the repository or download the code files.
2. Make sure the required libraries are installed (see Installation section).
3. Place the tumor data file (`tumor.csv`) in the same directory as the code files.
4. Run the code in a Python environment or Jupyter Notebook.

## Data

The tumor data is stored in a CSV file named `tumor.csv`. It contains information about different features of tumors and their corresponding classes (presence or absence of a tumor). The data is loaded into a pandas DataFrame and displayed using the `pandasgui` library, which provides an interactive interface for exploring the dataset.

## Exploratory Data Analysis

The code begins with an exploratory data analysis section, where various aspects of the dataset are examined. The following operations are performed:

- Displaying the loaded DataFrame using `pandasgui` to visualize the data.
- Describing the dataset to obtain summary statistics.
- Checking for missing values in the dataset.
- Dropping the 'Coarseness' column from the DataFrame due to its negligible average and no effect on the dataset.
- Creating a correlation matrix and generating a heatmap using seaborn to visualize the correlations between different features.

## Data Preprocessing

In the data preprocessing section, the dataset is split into input features (X) and the target variable (y). The 'Class' and 'Image' columns are excluded from the input features. The data is further divided into training and testing sets using a 90:10 ratio.

## Logistic Regression

A logistic regression model is created using a pipeline that includes standard scaling of the features and logistic regression as the classifier. Hyperparameter tuning is performed using grid search with cross-validation. The best hyperparameters are determined, and a new logistic regression model is trained with these optimal hyperparameters. The model is then evaluated using a classification report and a confusion matrix plot.

## Random Forest Classification

A random forest classifier model is created using a pipeline that includes standard scaling of the features and a random forest classifier. Hyperparameter tuning is performed using grid search with cross-validation. The best model obtained from the grid search is selected. The model is evaluated using a classification report and a confusion matrix plot.

## Results

The results of the logistic regression and random forest classification models are displayed using classification reports and confusion matrix plots. These results provide insights into the performance of the models in predicting tumor presence or absence.

## Conclusion

In this project, we successfully analyzed tumor data using logistic regression and random forest classification. The models were trained and evaluated on the dataset, providing classification reports and confusion matrix plots as performance metrics. The code and results presented in this README can serve as a guide for future analysis or improvements in the field of tumor classification.
