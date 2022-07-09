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
from sklearn.utils import shuffle
import os
import shutil, glob


for i in range (0,100):
# We repeat the process x number of times so that every PFAS falls at least once in the testing set
    def read_dataset():
        # Read datasets
        dataset = pd.read_csv('Rcm_database.csv')
        dataset = dataset.set_index('INCHIKEY')
        dataset = dataset.select_dtypes([np.number])
        dataset = dataset.reset_index()
        dataset['Rcm non-lipid normalized'] = dataset['Rcm non-lipid normalized'].replace(0, 0.01)
        dataset['logRcm'] = np.log10(dataset['Rcm non-lipid normalized'])
        return(dataset)
    
    dataset = read_dataset()
    
    sns.displot(dataset, x='logRcm',kind ='kde')
    sns.displot(dataset, x='logRcm')
    print('Reading dataset completed')
    
    def assign_and_split():
        # Assign X and y variables
        X1 = dataset.loc[:, 'ABC':'mZagreb2']
        X2 = dataset.loc[:, 'INCHIKEY']
        X = pd.concat([X1, X2], axis=1)
        y = dataset.loc[:, 'logRcm']
        y = shuffle(y, random_state=1) # y-randomization step -  use only for y-randomization analyis
        # Split dataframe into training and testing set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
        
        tr_inch = X_train.loc[:, 'INCHIKEY'].reset_index(drop=True)
        ts_inch = X_test.loc[:, 'INCHIKEY'].reset_index(drop=True)
        
        X_train = X_train.drop(['INCHIKEY'], axis=1)
        X_test = X_test.drop(['INCHIKEY'], axis=1)
        
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        return(X_train, X_test, y_train, y_test, tr_inch, ts_inch)
    
    X_train, X_test, y_train, y_test, tr_inch, ts_inch = assign_and_split()
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
    
      optimizer = tf.optimizers.Adam(0.001)
    
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
    
    EPOCHS = 10
    
    # The patience parameter is the amount of epochs to check for improvement
    #early_stop = keras.callbacks.EarlyStopping(monitor='val_loss')
    
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size = 50,
                        validation_split=0.2, verbose=0,
                        callbacks=[PrintDot()])
    
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
    df = pd.concat([df, tr_inch], axis=1)
    df.to_csv('ANN_yrand_tr{}'.format(i) + '.csv') # Print data to a csv file
    
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
    df = pd.concat([df, ts_inch], axis=1)
    df.to_csv('ANN_yrand_ts{}'.format(i) + '.csv') # Print data to a csv file # Print dataframe
    
    R2, mae = r_and_mae(y_test, X_test)
    
    print("Testing set R2 = " + str(R2))
    print("Testing set Mean Abs Error = " + str(mae))


# organizing output files
newpath = r'C:\Users\dimit\Dropbox\UCSF postdoc\Adi internship v2\Maternal - cord predictions\ANN_CV_yrandom1' 
if not os.path.exists(newpath):
    os.makedirs(newpath)
else:
    shutil.rmtree(newpath)
    os.makedirs(newpath)

source = 'C:/Users/dimit/Dropbox/UCSF postdoc/Adi internship v2/Maternal - cord predictions'
source_ls = os.listdir(source)
destination = 'C:/Users/dimit/Dropbox/UCSF postdoc/Adi internship v2/Maternal - cord predictions/ANN_CV_yrandom1'

for file in source_ls:
    if 'ANN_yrand' in file:
        shutil.move(file, destination)

os.chdir(destination)
extension = 'csv'
all_filenames = [i for i in glob.glob('ANN_yrand_tr*.{}'.format(extension))]
combined_tr = pd.concat([pd.read_csv(f) for f in all_filenames ])
combined_tr.to_csv("ANN_yrand_tr.csv", encoding='utf-8-sig')

all_filenames = [i for i in glob.glob('ANN_yrand_ts*.{}'.format(extension))]
combined_ts = pd.concat([pd.read_csv(f) for f in all_filenames ])
combined_ts.to_csv("ANN_yrand_ts.csv", encoding='utf-8-sig')


# Group chemicals in training and testing sets by INCHIKEY and calculate average values for Rcm
grouped_tr = combined_tr.groupby(by='INCHIKEY').mean()
grouped_tr = grouped_tr.drop(['Unnamed: 0'], axis=1)

grouped_ts = combined_ts.groupby(by='INCHIKEY').mean()
grouped_ts = grouped_ts.drop(['Unnamed: 0'], axis=1)

print(grouped_tr)
print(grouped_ts)

grouped_tr.to_csv('ANN_yrand_tr_grouped.csv')
grouped_ts.to_csv('ANN_yrand_ts_grouped.csv')
















