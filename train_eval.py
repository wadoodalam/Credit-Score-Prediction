import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier

def LoadData(path):
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
    
    # Concat the numeric and OH-encoded features as X-data
    X = pd.concat([data[numeric_features],encoded_features],axis=1)
    #Y-cols
    Y = data['Credit_Score']
    #Group for group-wise cross validation
    #group = data['Customer_ID']
    return X,Y

def TrainTestSplit(X,Y):
    # 80% train and 20% test
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.8,random_state=0)
    return X_train,X_test,Y_train,Y_test

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

if __name__ == "__main__":
    X,Y = LoadData('10kData.csv')
    # Split with 80/20
    X_train,X_test,Y_train,Y_test = TrainTestSplit(X,Y)

    GP = TrainModel(X_train,Y_train,GaussianProcessClassifier(random_state=0)) 
    
    accuracy, conf_matrix = Predict(X_test,Y_test, GP)
    
    print(accuracy)