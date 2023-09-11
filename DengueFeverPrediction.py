#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tsfresh import extract_features
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
# In[2]:

train = pd.read_csv('DengAI_Predicting_Disease_Spread_Training_Data_Features.csv')
feat_train = pd.read_csv('DengAI_Predicting_Disease_Spread_Training_Data_Labels.csv')
test = pd.read_csv('DengAI_Predicting_Disease_Spread_Test_Data_Features.csv')


# In[3]:


train.sample(5)


# In[ ]:





# In[3]:


train.fillna(train.mean(), inplace=True)


# In[4]:


test.fillna(train.mean(), inplace=True)


# In[5]:


test.isnull().sum()


# In[6]:


data=pd.merge(train, feat_train)


# In[9]:


data.info


# In[9]:


# Calculating the correlation matrix

correlation = data.corr()

f, ax = plt.subplots(figsize=(25, 10))
sns.heatmap(correlation, annot=True, ax=ax)


# In[10]:


# Visualizing the relationship between features which show strong correlation

figure, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

sns.scatterplot(x='reanalysis_sat_precip_amt_mm', y='precipitation_amt_mm', data=data, ax=axes[0,0])
sns.scatterplot(x='reanalysis_specific_humidity_g_per_kg', y='reanalysis_dew_point_temp_k', data=data, ax=axes[0,1])
sns.scatterplot(x='reanalysis_tdtr_k', y='reanalysis_max_air_temp_k', data=data, ax=axes[1,0])
axes[1, 1].axis('off')

plt.suptitle('Graphs Indicating Strong Correlation')

plt.show()


# In[11]:


# Finding the correlation for the above columns

feature1 = data['reanalysis_specific_humidity_g_per_kg']
feature2 = data['reanalysis_dew_point_temp_k']

feature3 = data['reanalysis_sat_precip_amt_mm']
feature4 = data['precipitation_amt_mm']

feature5 = data['reanalysis_tdtr_k']
feature6 = data['reanalysis_max_air_temp_k']

# Calculate the correlation between the two columns
correlation_1 = feature1.corr(feature2)
correlation_2 = feature3.corr(feature4)
correlation_3 = feature5.corr(feature6)

print("Correlation between temperature columns:", correlation_1)
print("Correlation between temperature columns:", correlation_2)
print("Correlation between temperature columns:", correlation_3)


# In[12]:


# Dropping the columns

data = data.drop(['reanalysis_tdtr_k'], axis=1)
data


# In[13]:


# Standardizing numerical variables

num_cols = ['ndvi_ne', 'ndvi_nw','reanalysis_specific_humidity_g_per_kg','reanalysis_sat_precip_amt_mm', 'ndvi_se', 'ndvi_sw', 'reanalysis_air_temp_k', 
            'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k', 
            'reanalysis_min_air_temp_k', 'reanalysis_precip_amt_kg_per_m2', 'reanalysis_relative_humidity_percent',
            'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c', 'station_min_temp_c',
            'station_precip_mm']

data[num_cols] = (data[num_cols] - data[num_cols].mean()) / data[num_cols].std()
data


# In[14]:


print(data['total_cases'].describe())
data['total_cases'].mean() #box plot


# In[15]:


import pandas as pd

# Define the threshold value
threshold = 24

# Create a new binary column for outbreak
data['outbreak'] = data['total_cases'].apply(lambda x: 1 if x >= threshold else 0)

# Calculate the IQR
Q1 = data['total_cases'].quantile(0.25)
Q3 = data['total_cases'].quantile(0.75)
IQR = Q3 - Q1

# Remove the outliers
data = data[(data['total_cases'] >= Q1 - 1.5 * IQR) & (data['total_cases'] <= Q3 + 1.5 * IQR)]

# Check the summary statistics of the total_cases column after removing the outliers
print(data['total_cases'].describe())


# In[16]:


exo_var = ['reanalysis_dew_point_temp_k', 'reanalysis_min_air_temp_k']
train_data_sa = data.loc[:1200, ['total_cases']+exo_var]
len(train_data_sa)-1


# In[86]:


#Edit code

# for 12

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Adding exogeneous variables
exo_var = ['reanalysis_dew_point_temp_k', 'reanalysis_min_air_temp_k','reanalysis_specific_humidity_g_per_kg']

# Split the data into training and testing sets
train_data_sa = data.loc[:700, ['total_cases']+exo_var]
test_data_sa = data.loc[701:, ['total_cases']+exo_var]

# Create a TimeSeriesSplit object
tscv = TimeSeriesSplit(n_splits=5)

# Train a SARIMAX model on the training data
sarimax_model = SARIMAX(train_data_sa['total_cases'], exog=train_data_sa[exo_var], order=(2, 1, 1), seasonal_order=(3, 1, 1, 12))
sarimax_results = sarimax_model.fit()

# Predict total_cases for the testing data
predictions = sarimax_results.predict(start=len(train_data_sa), end=len(data)-1, exog=test_data_sa[exo_var])

# Calculate mean absolute error of the predictions
mae = mean_absolute_error(test_data_sa['total_cases'], predictions)

print("Mean absolute error:", mae)


# In[94]:


#Edit code

# Plot actual and predicted values
plt.figure(figsize=(12, 6))
plt.plot(test_data_sa['total_cases'].reset_index(drop=True), label='Actual')
plt.plot(predictions.reset_index(drop=True), label='Predicted')
plt.legend()
plt.title('Actual vs. Predicted Dengue Cases')
plt.xlabel('Time (weeks)')
plt.ylabel('Total Cases')
plt.show()


# In[805]:


#Edit code

# for 24

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Adding exogeneous variables
exo_var = ['reanalysis_dew_point_temp_k', 'reanalysis_min_air_temp_k','reanalysis_specific_humidity_g_per_kg']

# Split the data into training and testing sets
train_data_sa = data.loc[:1200, ['total_cases']+exo_var]
test_data_sa = data.loc[1201:, ['total_cases']+exo_var]

# Create a TimeSeriesSplit object
tscv = TimeSeriesSplit(n_splits=5)

# Train a SARIMAX model on the training data
sarimax_model = SARIMAX(train_data_sa['total_cases'], exog=train_data_sa[exo_var], order=(2, 1, 1), seasonal_order=(3, 1, 1, 24))
sarimax_results = sarimax_model.fit()

# Predict total_cases for the testing data
predictions = sarimax_results.predict(start=len(train_data_sa), end=len(data)-1, exog=test_data_sa[exo_var])

# Calculate mean absolute error of the predictions
mae = mean_absolute_error(test_data_sa['total_cases'], predictions)

print("Mean absolute error:", mae)


# In[806]:


#Edit code

# Plot actual and predicted values
plt.figure(figsize=(12, 6))
plt.plot(test_data_sa['total_cases'].reset_index(drop=True), label='Actual')
plt.plot(predictions.reset_index(drop=True), label='Predicted')
plt.legend()
plt.title('Actual vs. Predicted Dengue Cases')
plt.xlabel('Time (weeks)')
plt.ylabel('Total Cases')
plt.show()


# In[809]:


#Edit code

# for 52

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Adding exogeneous variables
exo_var = ['reanalysis_dew_point_temp_k', 'reanalysis_min_air_temp_k','reanalysis_specific_humidity_g_per_kg']

# Split the data into training and testing sets
train_data_sa = data.loc[:1200, ['total_cases']+exo_var]
test_data_sa = data.loc[1201:, ['total_cases']+exo_var]

# Create a TimeSeriesSplit object
tscv = TimeSeriesSplit(n_splits=5)

# Train a SARIMAX model on the training data
sarimax_model = SARIMAX(train_data_sa['total_cases'], exog=train_data_sa[exo_var], order=(2, 1, 1), seasonal_order=(3, 1, 1, 52))
sarimax_results = sarimax_model.fit()

# Predict total_cases for the testing data
predictions = sarimax_results.predict(start=len(train_data_sa), end=len(data)-1, exog=test_data_sa[exo_var])

# Calculate mean absolute error of the predictions
mae = mean_absolute_error(test_data_sa['total_cases'], predictions)

print("Mean absolute error:", mae)

mse = mean_squared_error(test_data_sa['total_cases'], predictions)
rmse = np.sqrt(mse)

print("Root Mean Squared Error:", rmse)


# In[810]:


#Edit code

# Plot actual and predicted values
plt.figure(figsize=(12, 6))
plt.plot(test_data_sa['total_cases'].reset_index(drop=True), label='Actual')
plt.plot(predictions.reset_index(drop=True), label='Predicted')
plt.legend()
plt.title('Actual vs. Predicted Dengue Cases')
plt.xlabel('Time (weeks)')
plt.ylabel('Total Cases')
plt.show()


# In[30]:


#Edit code

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

#Define exogenous variables
exogenous_var = ['reanalysis_specific_humidity_g_per_kg', 'reanalysis_dew_point_temp_k', 'reanalysis_min_air_temp_k', 'station_min_temp_c']

#Split the data into training and testing sets
train_data_a = data.iloc[:1200]
test_data_a = data.iloc[1200:]

#Split the exogenous variables data into training and testing sets
train_exog_a = data[exogenous_var].iloc[:1200]
test_exog_a = data[exogenous_var].iloc[1200:]

#Perform differencing on the training data
train_diff = train_data_a['total_cases'].diff().dropna()

#Scale the training data
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_diff.values.reshape(-1, 1))

#Create a TimeSeriesSplit object
tscv = TimeSeriesSplit(n_splits=5)

#Train an ARIMA model on the training data with exogenous variables
arima_model = ARIMA(train_scaled, exog=train_exog_a.diff().iloc[1:], order=(1, 0, 0)) # Check if need to keep seasonal components
arima_results = arima_model.fit()

#Perform differencing on the testing data
test_diff = test_data_a['total_cases'].diff().dropna()

#Scale the testing data
test_scaled = scaler.transform(test_diff.values.reshape(-1, 1))

#Predict total_cases for the testing data with exogenous variables
predictions_diff_array = predictions_diff.values.reshape(-1, 1)
predictions = scaler.inverse_transform(predictions_diff_array).flatten()

#Calculate mean absolute error of the predictions
mae = mean_absolute_error(test_data_a['total_cases'].iloc[1:], predictions)

print("Mean absolute error:", mae)

mse = mean_squared_error(test_data_a['total_cases'].iloc[1:], predictions)
rmse = np.sqrt(mse)

print("Root Mean Squared Error:", rmse)

#Edit code

#Plot actual and predicted values
plt.figure(figsize=(12, 6))
plt.plot(test_data_a['total_cases'].reset_index(drop=True), label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.title('Actual vs. Predicted Dengue Cases')
plt.xlabel('Time (weeks)')
plt.ylabel('Total Cases')
plt.show()


# In[27]:


#Edit code

#Plot actual and predicted values
plt.figure(figsize=(12, 6))
plt.plot(test_data_a['total_cases'].reset_index(drop=True), label='Actual')
plt.plot(predictions.reset_index(drop=True), label='Predicted')
plt.legend()
plt.title('Actual vs. Predicted Dengue Cases')
plt.xlabel('Time (weeks)')
plt.ylabel('Total Cases')
plt.show()


# ## Setting manual threshold of 25 to binary classify the target variable

# In[111]:


threshold = 25

# create a new column with binary values based on the threshold
data['outbreak'] = data['total_cases'].apply(lambda x: 1 if x >= threshold else 0)
data['outbreak']


# In[112]:


data.iloc[:, -1]


# In[113]:


data.drop(['week_start_date'], axis=1, inplace=True)
data.drop(['city'], axis=1, inplace=True)
data2 = data.copy(deep=True)
data.drop(['total_cases'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-2], data.iloc[:, -1], test_size=0.2, random_state=42)


# In[114]:


data


# In[115]:


# train SVM model
svm = SVC(kernel="linear", C=1)
svm.fit(X_train, y_train)
svm_preds = svm.predict(X_test)

# train XGBoost model
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)

# draw confusion matrix
conf_mat = confusion_matrix(y_test, svm_preds)
print('Confusion Matrix:\n', conf_mat)

conf_mat = confusion_matrix(y_test, xgb_preds)
print('Confusion Matrix:\n', conf_mat)


# In[116]:


import matplotlib.pyplot as plt
import seaborn as sns

# draw confusion matrix
conf_mat_1 = confusion_matrix(y_test, svm_preds)

# create heatmap using seaborn
sns.heatmap(conf_mat_1, annot=True, cmap='Blues')

# set axis labels and title
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - SVM')

# display the plot
plt.show()


# In[117]:


import matplotlib.pyplot as plt
import seaborn as sns

# draw confusion matrix
conf_mat_2 = confusion_matrix(y_test, xgb_preds)

# create heatmap using seaborn
sns.heatmap(conf_mat_2, annot=True, cmap='Blues')

# set axis labels and title
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - XGBoost')

# display the plot
plt.show()


# In[118]:


# Check classification report for SVM
print("SVM Classification Report:")
print(classification_report(y_test, svm_preds))

# Check classification report for Random Forest
print("XGBoost Classification Report:")
print(classification_report(y_test, xgb_preds))


# In[ ]:





# ## Converting continuos variable into binary for binary classification using ROC curve analysis and SVM

# In[169]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc

# Define the threshold value
threshold = 24

# Create a new binary column for outbreak
data2['outbreak'] = data2['total_cases'].apply(lambda x: 1 if x >= threshold else 0)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data2.iloc[:, 4:-1], data2.iloc[:, -1], test_size=0.3, random_state=42)

# Train a classification model
svm = SVC(kernel='sigmoid', C=0.1, probability=True)
svm.fit(X_train, y_train)

# Predict the probability scores for the test set
proba_scores = svm.predict_proba(X_test)[:, 1]

# Calculate the ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, proba_scores)
roc_auc = auc(fpr, tpr)

# Find the optimal threshold value
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Create a new binary column based on the optimal threshold
test_proba_scores = svm.predict_proba(X_test)[:, 1]
y_pred = [1 if test_proba_scores[i] >= optimal_threshold else 0 for i in range(len(test_proba_scores))]

# Evaluate the performance of the model
conf_mat = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', conf_mat)
print('Optimal Threshold:', optimal_threshold)
print('AUC:', roc_auc)


# In[170]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create a confusion matrix heatmap
conf_mat_heatmap = sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')

# Set the axis labels and title
conf_mat_heatmap.set_xlabel('Predicted labels')
conf_mat_heatmap.set_ylabel('True labels')
conf_mat_heatmap.set_title('Confusion Matrix')

# Show the plot
plt.show()


# In[171]:


import matplotlib.pyplot as plt

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[172]:


from sklearn.utils.class_weight import compute_class_weight

# Define the threshold value
threshold = 24

# Create a new binary column for outbreak
data2['outbreak'] = data2['total_cases'].apply(lambda x: 1 if x >= threshold else 0)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data2.iloc[:, 4:-1], data2.iloc[:, -1], test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Get class weights
class_weights = dict(zip(np.unique(y_train), compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)))

# Train a classification model with balanced class weights
svm = SVC(kernel='sigmoid', C=0.1, probability=True, class_weight=class_weights)
svm.fit(X_train_scaled, y_train)

# Predict the probability scores for the test set
proba_scores = svm.predict_proba(X_test_scaled)[:, 1]

# Calculate the ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, proba_scores)
roc_auc = auc(fpr, tpr)

# Find the optimal threshold value
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Create a new binary column based on the optimal threshold
test_proba_scores = svm.predict_proba(X_test_scaled)[:, 1]
y_pred = [1 if test_proba_scores[i] >= optimal_threshold else 0 for i in range(len(test_proba_scores))]

# Evaluate the performance of the model
conf_mat = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', conf_mat)
print('Optimal Threshold:', optimal_threshold)
print('AUC:', roc_auc)


# In[173]:


# Check for Overfitting

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data2.iloc[:, 4:-1], data2.iloc[:, -1], test_size=0.3, random_state=42)

# Get class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(zip(np.unique(y_train), class_weights))

# Train a classification model with balanced class weights
svm = SVC(kernel='sigmoid', C=0.1, probability=True, class_weight=class_weights)
svm.fit(X_train, y_train)

# Predict the probability scores for the training set
proba_scores_train = svm.predict_proba(X_train)[:, 1]

# Calculate the ROC curve and AUC for the training set
fpr_train, tpr_train, thresholds_train = roc_curve(y_train, proba_scores_train)
roc_auc_train = auc(fpr_train, tpr_train)

# Predict the probability scores for the test set
proba_scores_test = svm.predict_proba(X_test)[:, 1]

# Calculate the ROC curve and AUC for the test set
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, proba_scores_test)
roc_auc_test = auc(fpr_test, tpr_test)

# Compare AUC scores for training and test sets
print('AUC score for training set:', roc_auc_train)
print('AUC score for test set:', roc_auc_test)


# In[174]:


from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(estimator=svm,
                                                        X=X_train_scaled,
                                                        y=y_train,
                                                        train_sizes=np.linspace(0.1, 1.0, 10),
                                                        cv=5,
                                                        n_jobs=-1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()


# In[175]:


from sklearn.model_selection import validation_curve

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(estimator=svm,
                                             X=X_train_scaled,
                                             y=y_train,
                                             param_name='C',
                                             param_range=param_range,
                                             cv=5,
                                             n_jobs=-1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.xscale('log')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()


# In[158]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create a confusion matrix heatmap
conf_mat_heatmap = sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')

# Set the axis labels and title
conf_mat_heatmap.set_xlabel('Predicted labels')
conf_mat_heatmap.set_ylabel('True labels')
conf_mat_heatmap.set_title('Confusion Matrix')

# Show the plot
plt.show()


# In[159]:


import matplotlib.pyplot as plt

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[31]:


from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf

# Define the model architecture
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model to the training data
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the performance of the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)


# In[515]:


# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# Calculate precision, recall, and F1 score
from sklearn.metrics import precision_score, recall_score, f1_score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1_score = f1_score(y_test, y_pred)

# Print the results
print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1_score)


# In[517]:


# Model Overfits after 15 epochs

# Fit the model to the training data
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Save the training and validation accuracy in lists
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Plot the training and validation accuracy over the epochs
import matplotlib.pyplot as plt

epochs = range(1, len(train_acc) + 1)
plt.plot(epochs, train_acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[32]:


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

# Define the model architecture
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Fit the model to the training data
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the performance of the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)


# In[33]:


from keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])


# In[34]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[35]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the data
train_features = pd.read_csv('DengAI_Predicting_Disease_Spread_Training_Data_Features.csv', index_col=[0,1,2])
train_labels = pd.read_csv('DengAI_Predicting_Disease_Spread_Training_Data_Labels.csv', index_col=[0,1,2])
test_features = pd.read_csv('DengAI_Predicting_Disease_Spread_Test_Data_Features.csv', index_col=[0,1,2])

test_features.drop(['week_start_date'] ,axis=1,inplace=True)
train_features.drop(['week_start_date'],axis=1,inplace=True)

# Create an imputer object with mean strategy
imputer = SimpleImputer(strategy='mean')

# Impute missing values in train_features and test_features
train_features = imputer.fit_transform(train_features)
test_features = imputer.transform(test_features)


# In[36]:


# Preprocess the data
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)



# Split the data into train and validation sets
split = int(0.8 * len(train_features))
X_train, y_train = train_features[:split], train_labels.total_cases.values[:split]
X_val, y_val = train_features[split:], train_labels.total_cases.values[split:]

# Define the KNN algorithm
class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        y_pred = []
        for x in X:
            distances = []
            for i, x_train in enumerate(self.X):
                distance = np.linalg.norm(x - x_train)
                distances.append((distance, i))
            distances = sorted(distances)[:self.k]
            k_indices = [i for _, i in distances]
            k_labels = self.y[k_indices]
            y_pred.append(np.mean(k_labels))
        return y_pred

# Train and evaluate the KNN algorithm
knn = KNN(k=15)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_val)
mse = np.mean((y_pred - y_val) ** 2)
print(f'Mean Squared Error: {mse:.2f}')
accuracy = np.mean(y_pred == y_val)
print(f'Accuracy: {accuracy:.2f}')


# In[37]:


from sklearn.decomposition import PCA

# Preprocess the data
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# Reduce the dimensionality of the data with PCA
pca = PCA(n_components=10)
train_features = pca.fit_transform(train_features)
test_features = pca.transform(test_features)

# Create an imputer object with mean strategy
imputer = SimpleImputer(strategy='mean')

# Impute missing values in train_features and test_features
train_features = imputer.fit_transform(train_features)
test_features = imputer.transform(test_features)

# Split the data into train and validation sets
split = int(0.8 * len(train_features))
X_train, y_train = train_features[:split], train_labels.total_cases.values[:split]
X_val, y_val = train_features[split:], train_labels.total_cases.values[split:]

# Define the KNN algorithm
class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        y_pred = []
        for x in X:
            distances = []
            for i, x_train in enumerate(self.X):
                distance = np.linalg.norm(x - x_train)
                distances.append((distance, i))
            distances = sorted(distances)[:self.k]
            k_indices = [i for _, i in distances]
            k_labels = self.y[k_indices]
            y_pred.append(np.mean(k_labels))
        return y_pred

# Train and evaluate the KNN algorithm
knn = KNN(k=15)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_val)
mse = np.mean((y_pred - y_val) ** 2)
print(f'Mean Squared Error: {mse:.2f}')
accuracy = np.mean(y_pred == y_val)
print(f'Accuracy: {accuracy:.2f}')


# In[ ]:





# In[63]:


len(train_features), train_labels.count()


# In[64]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

# Load the data
train_features = pd.read_csv('DengAI_Predicting_Disease_Spread_Training_Data_Features.csv', index_col=[0,1,2])
train_labels = pd.read_csv('DengAI_Predicting_Disease_Spread_Training_Data_Labels.csv', index_col=[0,1,2])
test_features = pd.read_csv('DengAI_Predicting_Disease_Spread_Test_Data_Features.csv', index_col=[0,1,2])

test_features.drop(['week_start_date'] ,axis=1,inplace=True)
train_features.drop(['week_start_date'],axis=1,inplace=True)

# Create an imputer object with mean strategy
imputer = SimpleImputer(strategy='mean')

# Impute missing values in train_features and test_features
train_features = imputer.fit_transform(train_features)
test_features = imputer.transform(test_features)

# Preprocess the data
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# Split the data into train and validation sets
split = int(0.8 * len(train_features))
X_train, y_train = train_features[:split], train_labels.total_cases.iloc[:split].values.reshape(-1, 1)
X_val, y_val = train_features[split:], train_labels.total_cases.iloc[split:].values.reshape(-1, 1)

# Hyperparameter tuning
param_grid = {'n_neighbors': np.arange(1, 31)}
knn = KNeighborsRegressor()
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Train and evaluate the KNN algorithm with the best hyperparameters
best_k = grid_search.best_params_['n_neighbors']
knn = KNeighborsRegressor(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_val)
mse = np.mean((y_pred - y_val) ** 2)
print(f'Mean Squared Error: {mse:.2f}')
accuracy = knn.score(X_val, y_val)
print(f'Accuracy: {accuracy:.2f}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[40]:


# Load the data
df = pd.read_csv('DengAI_Predicting_Disease_Spread_Training_Data_Features.csv')
df_feat = pd.read_csv('DengAI_Predicting_Disease_Spread_Training_Data_Labels.csv')

df=pd.merge(df, df_feat)
df


# In[48]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Fill missing values with the mean
data.fillna(data.mean(), inplace=True)

# Define the feature matrix X and target variable y
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the cost function
def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    J = -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return J

# Define the gradient descent function
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        h = sigmoid(np.dot(X, theta))
        theta = theta - (alpha / m) * np.dot(X.T, (h - y))
        J_history[i] = cost_function(X, y, theta)
    return theta, J_history

# Set the learning rate and number of iterations
alpha = 1
num_iters = 10000

# Add a column of ones to X for the intercept term
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

# Initialize the parameters theta
theta = np.zeros(X_train.shape[1])

# Run gradient descent to obtain the optimal parameters
theta, J_history = gradient_descent(X_train, y_train, theta, alpha, num_iters)

# Add a column of ones to X_test for the intercept term
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# Predict the test set and calculate the accuracy
y_pred = sigmoid(np.dot(X_test, theta))
y_pred = np.round(y_pred)
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)


# In[66]:


from sklearn.linear_model import LogisticRegression

#Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create an instance of the LogisticRegression class
lr = LogisticRegression()

#Fit the model on the training data
lr.fit(X_train, y_train)

#Predict the test set and calculate the accuracy
y_pred = lr.predict(X_test)
accuracy = lr.score(X_test, y_test)
print("Accuracy:", accuracy)


# In[ ]:




