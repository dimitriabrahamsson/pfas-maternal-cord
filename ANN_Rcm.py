#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 09:59:36 2022

@author: dimitriabrahamsson
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


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

# Compile the ANN model
def build_model():
  model = keras.Sequential([
    keras.layers.Flatten(input_shape=(X_train.shape[1],)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(100, activation='exponential'),
    keras.layers.Dense(1)
  ])

  optimizer = tf.optimizers.Adamax(0.001)

  model.compile(loss='mae',
                optimizer=optimizer,
                metrics=['mae'])
  return model

model = build_model()
model.summary()

print('Model compiled')

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 500

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=500)

history = model.fit(X_train, y_train, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])

print ('Model trained')

def plot_history(history):
    fig = plt.figure()
    ax = plt.subplot()
    ax.plot(history.epoch, np.array(history.history['mae']),
               label='Train Loss')
    ax.plot(history.epoch, np.array(history.history['val_mae']),
               label = 'Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.legend()
    plt.ylim([0, 2.5])
    #plt.xlim([0, 500])
    plt.show()
    return(fig)

fig = plot_history(history)    
fig.savefig('train_val_entact_500_20t.png', dpi=400)

# Use model to make predictions for the training set
y_pred = model.predict(X_train).flatten()

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
df.to_csv('ANN_plus_tr_1.csv') # Print data to a csv file

def r_and_mae(ytrue, Xtrue):
    corr_matrix = np.corrcoef(ytrue, y_pred)
    R = corr_matrix[0,1]
    R2 = np.round(R**2, 2)
    [loss, mae] = model.evaluate(Xtrue, ytrue, verbose=0)
    mae = np.round(mae, 2)
    return(R2, mae)

R2, mae = r_and_mae(y_train, X_train)

print("Training set R2 = " + str(R2))
print("Training set Mean Abs Error = " + str(mae))


# Use model to make predictions for the testing set
y_pred = model.predict(X_test).flatten()

fig = plot_trueVSpred(y_test)
fig = plot_error(y_test)

# Export data into a dataframe
df = pd.DataFrame({'true_values':y_test,'predicted_values':y_pred})
df.to_csv('ANN_plus_ts_1.csv') # Print dataframe

R2, mae = r_and_mae(y_test, X_test)

print("Testing set R2 = " + str(R2))
print("Testing set Mean Abs Error = " + str(mae))





























