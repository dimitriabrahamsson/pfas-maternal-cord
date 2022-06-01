# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 11:55:48 2022

@author: dimitriabrahamsson
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor as RFR
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

clf = GridSearchCV(RFR(), {
    'n_estimators': [10, 20, 100, 200, 500], 'random_state': [0, 1, 2, 5, 10]
    }, cv=5, scoring='neg_mean_absolute_error', return_train_score=False,
    verbose=5, n_jobs=-1)

clf.fit(X, y)

clf.cv_results_

df_cv = pd.DataFrame(clf.cv_results_)
print(df_cv)
df_cv.to_csv('RF_GridSearch4.csv')










