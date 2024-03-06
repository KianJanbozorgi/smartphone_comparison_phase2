# Advanced Clustering Techniques Notebook

## Overview

This Jupyter notebook, titled "clustering", is focused on the exploration of advanced clustering techniques and outlier detection methods. It aims to demonstrate the application of various algorithms for identifying patterns and anomalies within datasets, thereby facilitating a deeper understanding of the underlying data structures.

## Purpose

The primary purpose of this notebook is to:

- Conduct in-depth data exploration to uncover insights from the dataset.
- Apply advanced clustering techniques to segment the data into meaningful groups.
- Utilize outlier detection methods to identify and handle anomalies within the data.
- Evaluate the effectiveness of different clustering and outlier detection algorithms through metrics like the silhouette score.

## Technologies and Libraries Used

- **Python**: The core programming language used for analysis and modeling.
- **Pandas** and **NumPy**: For data manipulation and numerical operations.
- **Matplotlib**, **Seaborn**, and **Plotly**: For data visualization and interactive plots.
- **Scikit-learn**: Provides various clustering algorithms, preprocessing methods, and metrics for evaluation.
- **Yellowbrick**: For visualization of model selection and evaluation metrics.
- **IsolationForest**: An ensemble algorithm for efficient outlier detection.

## Key Features

- `OutlierDetector` class: Custom implementation for detecting outliers in the dataset.
- `optimise_k_means` function: A utility to find the optimal number of clusters for KMeans algorithm.

# Classification Analysis Notebook

## Overview

This Jupyter notebook, "classification.ipynb", delves into various classification techniques, demonstrating the application of machine learning models to perform classification tasks. It combines data exploration, preprocessing, model training, and evaluation to provide insights into effective classification strategies.

## Purpose

The notebook serves to:
- Perform comprehensive data exploration to understand the features and target variables within the dataset.
- Apply multiple classification models to the data, comparing their performance to identify the most effective approach for the given classification task.
- Evaluate model performance using metrics such as confusion matrix, F1 score, and accuracy.

## Technologies and Libraries Used

- **Python**: The primary programming language.
- **Pandas** and **NumPy**: For data manipulation and numerical calculations.
- **Matplotlib** and **Seaborn**: For data visualization.
- **Scikit-learn**: For data preprocessing, model training, and evaluation.
- **Regular Expressions (re)**: For text and data manipulation.

## Key Components

- `ModelTrainer` class: A utility for training and evaluating machine learning models.
- Custom functions for data preprocessing and evaluation

# Regression Analysis Notebook

## Overview

This Jupyter notebook, titled "regression", focuses on comprehensive regression analysis, including pre-processing, feature engineering, and the application of various regression models. The notebook aims to demonstrate effective strategies for predicting continuous outcomes based on given features.

## Purpose

The notebook is designed to:
- Explore and visualize relationships within the dataset to inform pre-processing and feature engineering strategies.
- Implement a variety of regression models, including linear regression, Lasso, and RandomForestRegressor, to compare their performance.
- Evaluate the models' performance using metrics such as mean absolute error, mean squared error, and R2 score.
- Utilize a custom class, `RegressionModelGridSearch`, for optimizing model parameters and enhancing predictive accuracy.

## Technologies and Libraries Used

- **Python**: For general programming.
- **Pandas** and **NumPy**: For data manipulation and numerical computations.
- **Matplotlib**, **Seaborn**, and **Plotly**: For data visualization.
- **Scikit-learn**: For machine learning models, data pre-processing, and performance evaluation.
- **LightGBM**: For gradient boosting framework that uses tree-based learning algorithms.

## Key Components

- Custom functions like `convert_to_average` and `categorize_display_type` for data pre-processing.
- `RegressionModelGridSearch` class for systematic model selection and hyperparameter tuning.
- Comprehensive use of regression models and evaluation metrics to establish best practices for regression analysis.

## Contributors

- [MahsaNouriZonouz](https://github.com/MahsaNouriZonouz)
- [Kian Janbozorgi](https://github.com/KianJanbozorgi)
- [Sanaz Gheibuni](https://github.com/sanaazz)
