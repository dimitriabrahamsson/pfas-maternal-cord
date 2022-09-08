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
        dataset_in = dataset.loc[:, 'INCHIKEY']
        dataset['Rcm non-lipid normalized'] = dataset['Rcm non-lipid normalized'].replace(0, 0.01)
        dataset['logRcm'] = np.log10(dataset['Rcm non-lipid normalized'])
        datasetR1 = dataset.loc[:, 'ABC':'mZagreb2']
        datasetR2 = dataset['logRcm']
        return(dataset, datasetR1, datasetR2, dataset_in)
    
    dataset, datasetR1, datasetR2, dataset_in = read_dataset()
    
    def read_pfas(dataset, datasetR1, datasetR2):
        pfas = pd.read_csv('df_mordred_PFAS_MasterList.csv', low_memory=False)
        pfas_in = pfas.loc[:, 'INCHIKEY']
        pfas = pfas.select_dtypes([np.number])
        pfas = pfas.loc[:, pfas.columns.isin(datasetR1.columns)]
        datasetR1 = datasetR1.loc[:, datasetR1.columns.isin(pfas.columns)]
        pfas = pfas.reindex(sorted(pfas.columns), axis=1)
        datasetR1 = datasetR1.reindex(sorted(datasetR1.columns), axis=1)
        dataset = pd.concat([dataset_in, datasetR1, datasetR2], axis=1)
        pfas = pd.concat([pfas_in, pfas], axis=1)
        pfas = pfas.drop_duplicates(keep='first')
        return(dataset, pfas)
    
    dataset, pfas = read_pfas(dataset, datasetR1, datasetR2)
    
    print(dataset)
    print(pfas)
        
    
    
    sns.displot(dataset, x='logRcm',kind ='kde')
    sns.displot(dataset, x='logRcm')
    print('Reading dataset completed')
    
    def assign_and_split():
    # Assign X and y variables
        X1 = dataset.loc[:, 'AATS0Z':'piPC9']
        X2 = dataset.loc[:, 'INCHIKEY']
        X = pd.concat([X1, X2], axis=1)
        y = dataset.loc[:, 'logRcm']
        # Split dataframe into training and testing set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
        
        tr_inch = X_train.loc[:, 'INCHIKEY']
        ts_inch = X_test.loc[:, 'INCHIKEY']
        pfas_in = pfas.loc[:,'INCHIKEY']
        
        X_train = X_train.drop(['INCHIKEY'], axis=1)
        X_test = X_test.drop(['INCHIKEY'], axis=1)
        F_test = pfas.drop(['INCHIKEY'], axis=1)
        
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        F_test = sc.transform(F_test)
        return(X_train, X_test, y_train, y_test, tr_inch, ts_inch, pfas_in, F_test)
    
    X_train, X_test, y_train, y_test, tr_inch, ts_inch, pfas_in, F_test = assign_and_split()
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
    #df.to_csv('ANN_all_PFAS_tr_{}'.format(i) + '.csv') # Print data to a csv file
    
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
    #df.to_csv('ANN_all_PFAS_ts_{}'.format(i) + '.csv') # Print data to a csv file # Print dataframe
    
    R2, mae = r_and_mae(y_test, X_test)
    
    print("Testing set R2 = " + str(R2))
    print("Testing set Mean Abs Error = " + str(mae))
    
    
    # Use model to make predictions for the testing set
    y_pred = model.predict(F_test).flatten()
    df = pd.DataFrame({'predicted_values':y_pred})
    df = pd.concat([pfas_in, df], axis=1)
    df = df.replace(-np.inf, np.NaN)
    df[df['predicted_values'] < -3] = np.NaN
    df[df['predicted_values'] > 3] = np.NaN
    df.to_csv('pfas_predictions_{}'.format(i) + '.csv')
    
    #sns.displot(df, x='predicted_values',kind ='kde')
    #sns.displot(df, x='predicted_values')


# organizing output files
newpath = r'C:\Users\dimit\Dropbox\UCSF postdoc\Adi internship v2\Maternal - cord predictions\PFAS_results3' 
if not os.path.exists(newpath):
    os.makedirs(newpath)
else:
    shutil.rmtree(newpath)
    os.makedirs(newpath)

source = 'C:/Users/dimit/Dropbox/UCSF postdoc/Adi internship v2/Maternal - cord predictions'
source_ls = os.listdir(source)
destination = 'C:/Users/dimit/Dropbox/UCSF postdoc/Adi internship v2/Maternal - cord predictions/PFAS_results3'

for file in source_ls:
    if 'pfas_predictions' in file:
        shutil.move(file, destination)

os.chdir(destination)
extension = 'csv'
all_filenames = [i for i in glob.glob('pfas_predictions_*.{}'.format(extension))]
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
combined_csv.to_csv("pfas_predictions_sum.csv", encoding='utf-8-sig')

grouped = combined_csv.groupby(by='INCHIKEY').mean()
grouped = grouped.drop(['Unnamed: 0'], axis=1)
print(grouped)

g = sns.displot(grouped, x='predicted_values',kind ='kde', color='dodgerblue')
plt.xlim(-4, 1)
g.savefig('pfas_distplot_1.png', dpi=300)


pfas_mean = grouped['predicted_values'].mean()
pfas_mean = np.round(pfas_mean, 3)
print('mean = ' + str(pfas_mean)) 

pfas_m = grouped[grouped['predicted_values'] < 0]
pfas_c = grouped[grouped['predicted_values'] > 0]
pfas_o = grouped[grouped['predicted_values'] == 0]
pfas_nan = grouped[grouped['predicted_values'] != grouped['predicted_values']]

len(pfas_m)
len(pfas_c)
len(pfas_o)
len(pfas_nan)


sns.displot(grouped, x='predicted_values', color='dodgerblue', alpha=0.5)
grouped.to_csv('pfas_grouped.csv')




















