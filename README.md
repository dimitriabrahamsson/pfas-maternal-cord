## Modeling the transplacental transfer of small molecules using machine learning models: A case study on PFAS
#### During pregnancy, chemicals from the mother are transfered to the fetus. This is often described as the transplacental transfer (TPT) or the partitioning of small molecules between maternal and cord blood (Rcm). Rcm is determined by the physicochemical properties of the chemicals and by the properties of their environment, in this case, the placenta.

#### In this study we examined the application of 3 different machine learning algorithms at predicting Rcm based on the chemicals' physicochemical properties. 

#### The first model we worked on was an artificial neural network (ANN).
To run the model open the script 'ANN_Rcm.py' and read the database file 'Rcm_database.csv'. The script outputs the results for the training and testing sets as two seperate files: 'ANN_plus_tr_1.csv' and 'ANN_plus_ts_1.csv'. 

#### The second model we examined was a random forest (RF).
To run the model open the script 'RF_Rcm.py' and read the database file 'Rcm_database.csv'. The script outputs the results for the training and testing sets as two seperate files: 'RF_plus_tr_1.csv' and 'RF_plus_ts_1.csv'. 

#### The third model was a support vector machine (SVM).
To run the model open the script 'SVM_Rcm.py' and read the database file 'Rcm_database.csv'. The script outputs the results for the training and testing sets as two seperate files: 'SVM_plus_tr_1.csv' and 'SVM_plus_ts_1.csv'. 
