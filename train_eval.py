import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split,StratifiedGroupKFold
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.stats.mstats import winsorize

def LoadData(path, Winsorize=False):
    # read from csv
    data = pd.read_csv(path)
    # feature with numeric values
    numeric_features = ['Age','Annual_Income','Monthly_Inhand_Salary','Num_Bank_Accounts','Num_Credit_Card',
                        'Interest_Rate','Num_of_Loan','Delay_from_due_date','Num_of_Delayed_Payment',
                        'Changed_Credit_Limit','Num_Credit_Inquiries','Outstanding_Debt',
                        'Credit_Utilization_Ratio','Total_EMI_per_month','Amount_invested_monthly',
                        'Monthly_Balance','Payment_of_Min_Amount']
    # Features with categorical variables
    categorical_features = ['Month','Occupation','Credit_Mix','Payment_Behaviour']
    
    # One Hot encoding for cat-features for X-data
    encoded_features = pd.get_dummies(data[categorical_features], drop_first=True)
    
    #Y-cols
    Y = data['Credit_Score']
    #Group for group-wise cross validation
    group = data['Customer_ID']
    
    if Winsorize:
        for feature in numeric_features:
            data[feature] = winsorize(data[feature],limits=[0.05,0.05])
        
    # Concat the numeric and OH-encoded features as X-data
    X = pd.concat([data[numeric_features],encoded_features],axis=1)
    return X,Y,group

def TrainModel(X,Y,classifier):
    # train the model
    classifier.fit(X,Y)
    return classifier

def Predict(X,Y,classifier):
    #  predict from the model
    Y_predict = classifier.predict(X)
    # get accuracy
    accuracy = accuracy_score(Y,Y_predict)
    # get conf matrix
    conf_matrix = confusion_matrix(Y, Y_predict)
    return accuracy, conf_matrix

def TrainTestSplit(X,Y,dummy=False):
    # 80% train and 20% test
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.8,random_state=0)
    if dummy:
        classifier = TrainModel(X_train,Y_train,DummyClassifier(strategy='stratified',random_state=0))
    else:
        classifier = TrainModel(X_train,Y_train,HistGradientBoostingClassifier(max_iter=100, random_state=0)) 
    accuracy, conf_matrix = Predict(X_test,Y_test, classifier)
    return accuracy,conf_matrix

def SGKFoldAccuracy(X,Y,group,dummy=False):
    if dummy:
        classifier = DummyClassifier(strategy='stratified',random_state=0)
    else:
        classifier = HistGradientBoostingClassifier(max_iter=100, random_state=0)
        
    sgk_fold = StratifiedGroupKFold(n_splits=10,shuffle=True)
    scores = [] 
    conf_matrices=[]
    for train,test in sgk_fold.split(X,Y,group):
        # Train and test separation by the gk_fold index
        X_train, X_test = X.iloc[train],X.iloc[test]
        Y_train, Y_test = Y.iloc[train],Y.iloc[test]
        
        # Train the classifier and predict
        classifier.fit(X_train,Y_train)
        Y_predict = classifier.predict(X_test)
        
        # get accuracy scores as numpy
        scores.append(accuracy_score(Y_test,Y_predict))
        conf_matrices.append(confusion_matrix(Y_test, Y_predict))
        
        
    # avg of the accuracy scores
    accuracy = np.mean(scores)
    conf_matrix = np.sum(conf_matrices, axis=0)
    return accuracy, conf_matrix  

def OutlierStrategy(solutions,dummy):
    accuracies = {'Abstain':[],'Winsorize':[],'Remove By Threshold':[]}
    conf_matricies = {'Abstain':[],'Winsorize':[],'Remove By Threshold':[]}
    for sol in solutions:
        if sol == 'abstain':
            X,Y,group = LoadData('10kData.csv',False)
            
            accuracy_TTS, conf_matrix_TTS = TrainTestSplit(X,Y,dummy)
            accuracy_SGK, conf_matrix_SGK = SGKFoldAccuracy(X,Y,group)
            
            accuracies['Abstain'].append(accuracy_TTS)
            accuracies['Abstain'].append(accuracy_SGK)
            
            conf_matricies['Abstain'].append(conf_matrix_TTS)
            conf_matricies['Abstain'].append(conf_matrix_SGK)
            
        elif sol == 'winzorize':
            X,Y,group = LoadData('10kData.csv',True)
            accuracy_TTS, conf_matrix_TTS = TrainTestSplit(X,Y,dummy)
            accuracy_SGK, conf_matrix_SGK = SGKFoldAccuracy(X,Y,group)
            
            accuracies['Winsorize'].append(accuracy_TTS)
            accuracies['Winsorize'].append(accuracy_SGK)

            conf_matricies['Winsorize'].append(conf_matrix_TTS)
            conf_matricies['Winsorize'].append(conf_matrix_SGK)
            
    return accuracies,conf_matricies
        
            
if __name__ == "__main__":
    
    outlier_sol = ['abstain','winzorize']
    dummy = 1
    if dummy == 1:
        accuracies,conf_matricies = OutlierStrategy(outlier_sol,True)
        print(accuracies)
    else:
        accuracies,conf_matricies = OutlierStrategy(outlier_sol,False)
        print(accuracies)
        print(conf_matricies)
    
    