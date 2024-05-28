# Dengue-Fever-Prediction

## Overview

This project focuses on the implementation of a Dengue Fever Prediction system, which aims to predict the number of Dengue fever cases in two cities based on various factors such as weather information, population data, and past Dengue incidence data. The project leverages machine learning techniques to provide accurate predictions that can help in early warning and prevention measures.

## Key Steps for Analysis

**Step 1: Exploratory Data Analysis and Preprocessing**

**Exploratory Data Analysis (EDA):** Perform EDA on the DengAI dataset to understand its structure and characteristics.

**Data Preprocessing:** Prepare the data for machine learning models by addressing missing values and conducting outlier detection to ensure data quality.

**Step 2: SARIMAX Time Series Models**

**Model Implementation:** Implement Seasonal Autoregressive Integrated Moving Average with Exogenous Variables (SARIMAX) models with seasonal periods of 12, 24, and 52 weeks.

**Evaluation:** Calculate the mean absolute error (MAE) of the model predictions and visualize the actual versus predicted values.

**Step 3: Binary Classification Using SVM and XGBoost**

**Binary Classification:** Explore binary classification using Support Vector Machine (SVM) and XGBoost algorithms.

**Evaluation:** Set a threshold to classify outbreaks and evaluate model performance using confusion matrices and classification reports.

**Step 4: SVM Binary Classification and Overfitting Check**

**Performance Evaluation:** Evaluate SVM model performance using ROC curves, AUC, and confusion matrices.

**Overfitting Check:** Compare training and test set performance to check for overfitting.

**Step 5: Binary Classification Neural Network with Keras**

**Neural Network:** Build, train, and evaluate a binary classification neural network using the Keras API.

**Metrics Monitoring:** Monitor metrics such as precision, recall, and F1 score. Implement early stopping to prevent overfitting.

**Step 6: Implementation of KNN and Logistic Regression from Scratch**

**Algorithm Implementation:** Implement the K-Nearest Neighbors (KNN) and Logistic Regression algorithms from scratch in Python.

**Demonstration:** Demonstrate how these algorithms can be implemented without relying on external libraries.

## Technologies Used

**Programming Language:** Python

**Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Keras, Statsmodels

**Visualization:** Matplotlib, Seaborn

**Machine Learning Models:** SARIMAX, SVM, XGBoost, Neural Networks, KNN, Logistic Regression

## Contact

For any questions or further information, you can reach me at omkar.yeole@colorado.edu.

Thank you for visiting the Dengue Fever Prediction project!
