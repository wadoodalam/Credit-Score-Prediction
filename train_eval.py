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
    numeric_features = ['Age', 'Annual_Income','Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
                        'Num_Credit_Card',
                        'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
                        'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
                        'Num_Credit_Inquiries', 'Outstanding_Debt',
                        'Credit_Utilization_Ratio', 'Credit_History_Age',
                        'Total_EMI_per_month', 'Amount_invested_monthly',
                        'Monthly_Balance', 'Auto Loan', 'Payday Loan',
                        'Student Loan', 'Credit-Builder Loan', 'Mortgage Loan',
                        'Home Equity Loan', 'Not Specified', 'Debt Consolidation Loan',
                        'Personal Loan','Payment_of_Min_Amount']
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
    model = GetClassifier(dummy)

    classifier = TrainModel(X_train,Y_train,model)
    accuracy, conf_matrix = Predict(X_test,Y_test, classifier)
    return accuracy,conf_matrix

def SGKFoldAccuracy(X,Y,group,dummy=False):
    classifier = GetClassifier(dummy)
        
    sgk_fold = StratifiedGroupKFold(n_splits=10,shuffle=True,random_state=0)
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

def GetClassifier(dummy=False):
    if dummy:
        model = DummyClassifier(strategy='most_frequent',random_state=0)
    else:
        model = HistGradientBoostingClassifier(max_iter=100, random_state=0)
    return model

def IdentifyOutliers():
    data = pd.read_csv('10kData.csv')
    
    # Find 25%/75% quartiles
    lower_quartile = data.quantile(0.25, numeric_only=True)
    upper_quartile = data.quantile(0.75, numeric_only=True)

    # Calculate IQR 
    inter_quartile = upper_quartile - lower_quartile

    # Calculate UB & LB for outliers
    lower_bound = lower_quartile - 1.5 * inter_quartile
    upper_bound = upper_quartile + 1.5 * inter_quartile

    # Align dataFrame to prevent bug
    data_aligned, lower_bound_aligned = data.align(lower_bound, axis=1, join='inner')
    data_aligned, upper_bound_aligned = data.align(upper_bound, axis=1, join='inner')
    # Identify outliers
    outliers = data_aligned[(data_aligned < lower_bound_aligned) | (data_aligned > upper_bound_aligned)]

    # save to csv
    outliers.to_csv('outliers.csv')

def RemoveByThreshold():
        # read from csv
    data = pd.read_csv('10kData.csv')
    # feature with numeric values
    numeric_features = ['Age', 'Annual_Income','Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
                        'Num_Credit_Card',
                        'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
                        'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
                        'Num_Credit_Inquiries', 'Outstanding_Debt',
                        'Credit_Utilization_Ratio', 'Credit_History_Age',
                        'Total_EMI_per_month', 'Amount_invested_monthly',
                        'Monthly_Balance', 'Auto Loan', 'Payday Loan',
                        'Student Loan', 'Credit-Builder Loan', 'Mortgage Loan',
                        'Home Equity Loan', 'Not Specified', 'Debt Consolidation Loan',
                        'Personal Loan','Payment_of_Min_Amount']
    # Features with categorical variables
    categorical_features = ['Month','Occupation','Credit_Mix','Payment_Behaviour']
    
    # One Hot encoding for cat-features for X-data
    encoded_features = pd.get_dummies(data[categorical_features], drop_first=True)
    
    # Thresholds determined by domain knowledge and visualization
    age_limit = (0,100) # By realistic age of people
    num_bank_account_limit = (0,11) # Forbes avg and dataset evaluation
    num_credit_cards_limit = (0,50) # More flexible threshold due to no limits, but realistic domain knowledge
    interest_Rate_limit = (0,19) # Max ever recorded in US history
    num_of_Loan_limit = (0,10) # Max number of loans by domain knowledge
    num_of_delayed_payment_limit = (0,50) # Max delayed payments by domain knowledge
    Num_Credit_Inquiries_limit = (0,50) # Domain knowledge and dataset
    credit_history_age_limit = (1,100) # Cannot be more than max age limit
    
    # Replace outliers by threshold with null values
    data.loc[data['Age'] < age_limit[0], 'Age'] = np.nan
    data.loc[data['Age'] > age_limit[1], 'Age'] = np.nan

    data.loc[data['Num_Bank_Accounts'] < num_bank_account_limit[0], 'Num_Bank_Accounts'] = np.nan
    data.loc[data['Num_Bank_Accounts'] > num_bank_account_limit[1], 'Num_Bank_Accounts'] = np.nan

    data.loc[data['Num_Credit_Card'] < num_credit_cards_limit[0], 'Num_Credit_Card'] = np.nan
    data.loc[data['Num_Credit_Card'] > num_credit_cards_limit[1], 'Num_Credit_Card'] = np.nan

    data.loc[data['Interest_Rate'] < interest_Rate_limit[0], 'Interest_Rate'] = np.nan
    data.loc[data['Interest_Rate'] > interest_Rate_limit[1], 'Interest_Rate'] = np.nan

    data.loc[data['Num_of_Loan'] < num_of_Loan_limit[0], 'Num_of_Loan'] = np.nan
    data.loc[data['Num_of_Loan'] > num_of_Loan_limit[1], 'Num_of_Loan'] = np.nan

    data.loc[data['Num_of_Delayed_Payment'] < num_of_delayed_payment_limit[0], 'Num_of_Delayed_Payment'] = np.nan
    data.loc[data['Num_of_Delayed_Payment'] > num_of_delayed_payment_limit[1], 'Num_of_Delayed_Payment'] = np.nan

    data.loc[data['Num_Credit_Inquiries'] < Num_Credit_Inquiries_limit[0], 'Num_Credit_Inquiries'] = np.nan
    data.loc[data['Num_Credit_Inquiries'] > Num_Credit_Inquiries_limit[1], 'Num_Credit_Inquiries'] = np.nan

    data.loc[data['Credit_History_Age'] < credit_history_age_limit[0], 'Credit_History_Age'] = np.nan
    data.loc[data['Credit_History_Age'] > credit_history_age_limit[1], 'Credit_History_Age'] = np.nan

#
#
    # Concat the numeric and OH-encoded features as X-data
    X = pd.concat([data[numeric_features],encoded_features],axis=1)

    #Y-cols
    Y = data['Credit_Score']
    #Group for group-wise cross validation
    group = data['Customer_ID']
    
    return X,Y,group

def OutlierStrategy(solutions,dummy):
    accuracies = {'Do Nothing':[],'Winsorize':[],'Remove By Threshold':[]}
    conf_matricies = {'Do Nothing':[],'Winsorize':[],'Remove By Threshold':[]}
    for sol in solutions:
        if sol == 'Do Nothing':
            X,Y,group = LoadData('10kData.csv',False)
            
            accuracy_TTS, conf_matrix_TTS = TrainTestSplit(X,Y,dummy)
            accuracy_SGK, conf_matrix_SGK = SGKFoldAccuracy(X,Y,group,dummy)
            
            accuracies['Do Nothing'].append(accuracy_TTS)
            accuracies['Do Nothing'].append(accuracy_SGK)
            
            conf_matricies['Do Nothing'].append(conf_matrix_TTS)
            conf_matricies['Do Nothing'].append(conf_matrix_SGK)
            
        elif sol == 'Winsorize':
            X,Y,group = LoadData('10kData.csv',True)
            
            accuracy_TTS, conf_matrix_TTS = TrainTestSplit(X,Y,dummy)
            accuracy_SGK, conf_matrix_SGK = SGKFoldAccuracy(X,Y,group,dummy)
            
            accuracies['Winsorize'].append(accuracy_TTS)
            accuracies['Winsorize'].append(accuracy_SGK)

            conf_matricies['Winsorize'].append(conf_matrix_TTS)
            conf_matricies['Winsorize'].append(conf_matrix_SGK)
            
        elif sol == 'Remove By Threshold':
            # Save outliers to outliers.csv for visualization
            IdentifyOutliers()
            X,Y,group = RemoveByThreshold()
            
            accuracy_TTS, conf_matrix_TTS = TrainTestSplit(X,Y,dummy)
            accuracy_SGK, conf_matrix_SGK = SGKFoldAccuracy(X,Y,group,dummy)
            
            accuracies['Remove By Threshold'].append(accuracy_TTS)
            accuracies['Remove By Threshold'].append(accuracy_SGK)

            conf_matricies['Remove By Threshold'].append(conf_matrix_TTS)
            conf_matricies['Remove By Threshold'].append(conf_matrix_SGK)
            
                
            
    return accuracies,conf_matricies
        

    
if __name__ == "__main__":

    outlier_sol = ['Do Nothing','Winsorize','Remove By Threshold']
    dummy = 1
    if dummy == 1:
        accuracies,conf_matricies = OutlierStrategy(outlier_sol,True)
        print(accuracies)
    else:
        accuracies,conf_matricies = OutlierStrategy(outlier_sol,False)
        print(accuracies)
        #print(conf_matricies)
    
  
    

    