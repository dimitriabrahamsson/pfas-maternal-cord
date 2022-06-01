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
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV


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

# Assign X and y variables
X = dataset.loc[:, 'ABC':'mZagreb2'].values
y = dataset.loc[:, 'logRcm'].values

sc = StandardScaler()
X = sc.fit_transform(X)
print('Assigning and spliting completed')

# Compile the ANN model
def build_model(optimizer='adam', learning_rate=0.001):
  model = keras.Sequential([
    keras.layers.Flatten(input_shape=(X.shape[1],)),
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

 #optimizer = tf.optimizers.Adamax(0.001)

  model.compile(loss='mae',
                optimizer=optimizer,
                metrics=['mae'])
  return model

model = build_model()
model.summary()

model = KerasRegressor(build_fn=build_model, verbose=0)

epochs = [10, 50, 200]
batch_size = [10, 50, 200]
optimizer = ['Adam', 'Adamax', 'Adadelta']
learning_rate =[0.001, 0.0001, 0.00001]


param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer, learning_rate=learning_rate)
grid = GridSearchCV(estimator=model, param_grid=param_grid, 
                    scoring='neg_mean_absolute_error', return_train_score=False,
                    verbose=5, n_jobs=-1, cv=5)
grid_result = grid.fit(X, y)

grid.cv_results_
df_cv = pd.DataFrame(grid.cv_results_)
print(df_cv)

df_cv.to_csv('ANN_GridSearch1.csv')
































