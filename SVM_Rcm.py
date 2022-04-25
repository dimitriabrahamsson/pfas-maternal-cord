# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 13:18:22 2022

@author: dimitriabrahamsson
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn import metrics

def read_dataset():
    # Read datasets
    dataset = pd.read_csv('Rcm_database.csv')
    dataset = dataset.select_dtypes([np.number])
    dataset['Rcm non-lipid normalized'] = dataset['Rcm non-lipid normalized'].replace(0, 0.01)
    dataset['logRcm'] = np.log10(dataset['Rcm non-lipid normalized'])
    return(dataset)

dataset = read_dataset()

sns.displot(dataset, x='logRcm',kind ='kde')
sns.displot(dataset, x='logRcm')
print('Reading dataset completed')

def assign_and_split():
    # Assign X and y variables
    X = dataset.loc[:, 'ABC':'mZagreb2'].values
    y = dataset.loc[:, 'logRcm'].values
    # Split dataframe into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return(X_train, X_test, y_train, y_test)

X_train, X_test, y_train, y_test = assign_and_split()
print('Assigning and spliting completed')

regressor = SVR(kernel='rbf', epsilon =0.2, C=1)
regressor.fit(X_train, y_train.ravel())
print('Model compiled and trained')

# Use model to make predictions for training set
y_pred = regressor.predict(X_train).flatten()
print('Predictions made for training set')

# Calculate error statistics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_train, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))

def plot_trueVSpred(ytrue):
    fig = plt.figure()
    plt.scatter(ytrue, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    _ = plt.plot([-100, 100], [-100, 100])
    plt.show()
    return(fig)

fig = plot_trueVSpred(y_train)

def plot_error(ytrue):
    fig = plt.figure()
    error = y_pred - ytrue
    plt.hist(error, bins = 10)
    plt.xlabel("Prediction Error")
    _ = plt.ylabel("Count")
    plt.show()
    return(fig)

fig = plot_error(y_train)

# Export data into a dataframe
df = pd.DataFrame({'true_values':y_train,'predicted_values':y_pred})
df.to_csv('SVM_plus_tr_1.csv') # Print data to a csv file

def r_and_mae(ytrue, Xtrue):
    corr_matrix = np.corrcoef(ytrue, y_pred)
    R = corr_matrix[0,1]
    R2 = np.round(R**2, 2)
    mae = metrics.mean_absolute_error(ytrue, y_pred)
    mae = np.round(mae, 2)
    return(R2, mae)

R2, mae = r_and_mae(y_train, X_train)

print("Training set R2 = " + str(R2))
print("Training set Mean Abs Error = " + str(mae))


# Use model to make predictions for the testing set
y_pred = regressor.predict(X_test).flatten()

fig = plot_trueVSpred(y_test)
fig = plot_error(y_test)

# Export data into a dataframe
df = pd.DataFrame({'true_values':y_test,'predicted_values':y_pred})
df.to_csv('SVM_plus_ts_1.csv') # Print dataframe

R2, mae = r_and_mae(y_test, X_test)

print("Testing set R2 = " + str(R2))
print("Testing set Mean Abs Error = " + str(mae))









