# Dengue-Fever-Prediction

This repository contains the implementation of a Dengue Fever Prediction project, which aims to predict the number of Dengue fever cases in two cities based on various factors such as weather information, population data, and past Dengue incidence data. The project consists of several key steps, each of which is detailed below.

Step 1: Exploratory Data Analysis and Preprocessing

In this step, we perform exploratory data analysis (EDA) on the DengAI dataset to understand its structure and characteristics. We preprocess the data to prepare it for machine learning models and address missing values. We also conduct outlier detection to ensure data quality.

Step 2: SARIMAX Time Series Models

We implement Seasonal Autoregressive Integrated Moving Average with Exogenous Variables (SARIMAX) models with seasonal periods of 12, 24, and 52 weeks to predict the number of Dengue cases in subsequent weeks. We calculate the mean absolute error (MAE) of the model predictions and visualize the actual versus predicted values.

Step 3: Binary Classification Using SVM and XGBoost

In this step, we explore binary classification using Support Vector Machine (SVM) and XGBoost algorithms. We set a threshold to classify outbreaks and evaluate model performance using confusion matrices and classification reports.

Step 4: SVM Binary Classification and Overfitting Check

We delve deeper into SVM classification by evaluating model performance using ROC curves, AUC, and confusion matrices. Additionally, we check for overfitting by comparing training and test set performance.

Step 5: Binary Classification Neural Network with Keras

This step involves building, training, and evaluating a binary classification neural network using the Keras API. We monitor metrics such as precision, recall, and F1 score. The script also implements early stopping to prevent overfitting.

Step 6: Implementation of KNN and Logistic Regression from Scratch

The final step showcases the implementation of the K-Nearest Neighbors (KNN) and Logistic Regression algorithms from scratch in Python. It demonstrates how these algorithms can be implemented without relying on external libraries.
