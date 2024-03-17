'''
Author: Wadood Alam
Date: 6th March 2024
Class: AI 539
Final Project: Credit Score Evaluation
'''

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split,StratifiedGroupKFold
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.stats.mstats import winsorize
from sklearn.impute import SimpleImputer
import time
import warnings
warnings.filterwarnings('ignore')

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
    
    
    #Y-cols
    Y = data['Credit_Score']
    #Group for group-wise cross validation
    group = data['Customer_ID']
    
    # Winzorize the data if requested(for outliers)
    if Winsorize:
        for feature in numeric_features:
            # Lowest/highest 5% replaced 5th/95th percentile respectively
            data[feature] = winsorize(data[feature],limits=[0.05,0.05])
        
    
    return data[numeric_features],data[categorical_features],Y,group

def OneHotEncoding(num_features,cat_features):
    # One Hot encoding for categorical -features 
    encoded_features = pd.get_dummies(cat_features, drop_first=True)
    # Concat the numeric and OH-encoded features as X-data
    X = pd.concat([num_features,encoded_features],axis=1)
    return X

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
    # Get the classifier depending on the dummy flag
    model = GetClassifier(dummy)
    # Train the model using the method
    classifier = TrainModel(X_train,Y_train,model)
    # Predict to get accuracy and confusion matrix
    accuracy, conf_matrix = Predict(X_test,Y_test, classifier)
    return accuracy,conf_matrix

def SGKFoldAccuracy(X,Y,group,dummy=False):
    # Get classifier based on dummy
    classifier = GetClassifier(dummy)
    # Init Strified-Group-wise CV, scores list and conf matrix list    
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
    # If dummy flag is true than return dummy model with majority class strategy
    # Otherwise return HistGradientBoostingClassifier
    if dummy:
        model = DummyClassifier(strategy='most_frequent',random_state=0)
    else:
        model = HistGradientBoostingClassifier(max_iter=100, random_state=0)
        #print(model.get_params()) # Use this print model parameters
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


    # One Hot encoding for cat-features for X-data
    encoded_features = pd.get_dummies(data[categorical_features], drop_first=True)
    # Concat the numeric and OH-encoded features as X-data
    X = pd.concat([data[numeric_features],encoded_features],axis=1)

    #Y-cols
    Y = data['Credit_Score']
    #Group for group-wise cross validation
    group = data['Customer_ID']
    
    return X,Y,group

def OutlierStrategy(solutions,dummy):
    # Dicts init for metric storage
    accuracies = {'Do Nothing':[],'Winsorize':[],'Remove By Threshold':[]}
    conf_matricies = {'Do Nothing':[],'Winsorize':[],'Remove By Threshold':[]}
    runtime = {'Do Nothing':[],'Winsorize':[],'Remove By Threshold':[]}
    # Loop through each strategy for outliers
    for sol in solutions:
        # start/end time variables
        start_time = 0
        end_time = 0
        
        if sol == 'Do Nothing':
            # Get cat and num features with winzorize flag as false
            numeric_features,categorical_features,Y,group = LoadData('10kData.csv',False)
            # OH-encode categorical feature and concat with num features
            X = OneHotEncoding(numeric_features,categorical_features)
            # start train-eval time
            start_time = time.time()
            # get accuracy and conf matrix for both eval methods
            accuracy_TTS, conf_matrix_TTS = TrainTestSplit(X,Y,dummy)
            accuracy_SGK, conf_matrix_SGK = SGKFoldAccuracy(X,Y,group,dummy)
            # end train-eval time
            end_time = time.time()
            # append accuracy and confusion matrix to the respective dicts
            accuracies['Do Nothing'].append(accuracy_TTS)
            accuracies['Do Nothing'].append(accuracy_SGK)
            conf_matricies['Do Nothing'].append(conf_matrix_TTS)
            conf_matricies['Do Nothing'].append(conf_matrix_SGK)
            
            # calculate and append time take
            time_taken = (end_time-start_time) 
            runtime['Do Nothing'].append(time_taken)
            
        elif sol == 'Winsorize':
            # Get cat and num features with winzorize flag as True(winzorized data)
            numeric_features,categorical_features,Y,group = LoadData('10kData.csv',True)
            # OH-encode categorical feature and concat with num features
            X = OneHotEncoding(numeric_features,categorical_features)
            # start train-eval time                   
            start_time = time.time()
            # get accuracy and conf matrix for both eval methods
            accuracy_TTS, conf_matrix_TTS = TrainTestSplit(X,Y,dummy)
            accuracy_SGK, conf_matrix_SGK = SGKFoldAccuracy(X,Y,group,dummy)
            # end train-eval time
            end_time = time.time()
            #append accuracy and confusion matrix to the respective dicts
            accuracies['Winsorize'].append(accuracy_TTS)
            accuracies['Winsorize'].append(accuracy_SGK)
            conf_matricies['Winsorize'].append(conf_matrix_TTS)
            conf_matricies['Winsorize'].append(conf_matrix_SGK)
            
            # calculate and append time take
            time_taken = (end_time-start_time) 
            runtime['Winsorize'].append(time_taken)
            
        elif sol == 'Remove By Threshold':
            
            # Save outliers to outliers.csv for visualization
            IdentifyOutliers()
            # Get X,Y after threshold removal
            X,Y,group = RemoveByThreshold()
            # train-eval start time
            start_time = time.time()
            # get accuracy and conf matrix for both eval methods
            accuracy_TTS, conf_matrix_TTS = TrainTestSplit(X,Y,dummy)
            accuracy_SGK, conf_matrix_SGK = SGKFoldAccuracy(X,Y,group,dummy)
            # train-eval end time
            end_time = time.time()
            #append accuracy and confusion matrix to the respective dicts
            accuracies['Remove By Threshold'].append(accuracy_TTS)
            accuracies['Remove By Threshold'].append(accuracy_SGK)
            conf_matricies['Remove By Threshold'].append(conf_matrix_TTS)
            conf_matricies['Remove By Threshold'].append(conf_matrix_SGK)
            
            # calculate and append time take
            time_taken = (end_time-start_time) 
            runtime['Remove By Threshold'].append(time_taken)
            
    return accuracies,conf_matricies,runtime
        
def C1Results(dummy):
    outlier_sol = ['Do Nothing','Winsorize', 'Remove By Threshold']
    if dummy:
        accuracies,conf_matricies,runtime = OutlierStrategy(outlier_sol,True)
        print('Run Time:',runtime)
        print('Accuracy:',accuracies)
        
        
    else:
        accuracies,conf_matricies,runtime = OutlierStrategy(outlier_sol,False)
        print('Run Time:',runtime)
        print('Accuracy:',accuracies)
        print(conf_matricies)
    
    
    



def ImputeByGroup(numeric_features,categorical_features,group):
    # Define and init imputation strategies and feature type
    mean_impute_features =  ['Changed_Credit_Limit', 'Credit_History_Age', 'Amount_invested_monthly', 'Monthly_Balance','Num_of_Delayed_Payment']
    mode_impute_num_features = ['Monthly_Inhand_Salary','Num_of_Loan','Num_Credit_Inquiries','Payment_of_Min_Amount']
    mode_impute_cat_features = [ 'Occupation' , 'Credit_Mix', 'Payment_Behaviour']
    
    # Impute for each group(mean) for the defined numeric features
    for feature in mean_impute_features:
        # Fill all the Null values in the group with mean of the corresponding group(s)
        numeric_features[feature] = numeric_features[feature].fillna(numeric_features.groupby(group)[feature].transform('mean'))
    
    # Impute for each group(mode) for the defined numeric features
    for feature in mode_impute_num_features:
        # Fill all the Null values in the group with mode of the corresponding group(s)
        numeric_features[feature] = numeric_features[feature].fillna(numeric_features.groupby(group)[feature].transform(lambda x: x.mode()[0]))

    # Impute for each group(mode) for the defined categorical features
    for feature in mode_impute_cat_features:
        # Fill all the Null values in the group with mode of the corresponding group(s)
        categorical_features[feature] = categorical_features[feature].fillna(categorical_features.groupby(group)[feature].transform(lambda x: x.mode()[0]))
    
    return numeric_features,categorical_features

def ImputeByFeature(numeric_features,categorical_features):
    # Define and init imputation strategies and feature type
    mean_impute_features =  ['Changed_Credit_Limit', 'Credit_History_Age', 'Amount_invested_monthly', 'Monthly_Balance','Num_of_Delayed_Payment']
    mode_impute_num_features = ['Monthly_Inhand_Salary','Num_of_Loan','Num_Credit_Inquiries','Payment_of_Min_Amount']
    mode_impute_cat_features = [ 'Occupation' , 'Credit_Mix', 'Payment_Behaviour']
    
    
    # Imputation for mean numeric features
    for feature in mean_impute_features:
        numeric_features[feature].fillna(numeric_features[feature].mean(),inplace=True)
    
        # Imputation for mode numeric features
    for feature in mode_impute_num_features:
       numeric_features[feature].fillna(numeric_features[feature].mode()[0],inplace=True)

    for feature in mode_impute_cat_features:
        categorical_features[feature].fillna(categorical_features[feature].mode()[0],inplace=True)
    
    return numeric_features,categorical_features

def MissingDataStrategy(solutions,dummy=False):
    # Dicts init for metric storage
    accuracies =        {'Impute by Group':[],'Impute by Feature':[]}
    conf_matricies =    {'Impute by Group':[],'Impute by Feature':[]}
    runtime =           {'Impute by Group':[],'Impute by Feature':[]}
    
    for sol in solutions:
        start_time = 0
        end_time = 0
        if sol == 'Impute by Group':
            # Get cat and num features with winzorize flag as True(winzorized data)
            numeric_features,categorical_features,Y,group = LoadData('10kData.csv',True)
            # Impute by group(12 features)
            numeric_features,categorical_features = ImputeByGroup(numeric_features,categorical_features,group)
            # OH-encode categorical feature and concat with num features
            X = OneHotEncoding(numeric_features,categorical_features)
            
            # start train-eval time                   
            start_time = time.time()
            # get accuracy and conf matrix for both eval methods
            accuracy_TTS, conf_matrix_TTS = TrainTestSplit(X,Y,dummy)
            accuracy_SGK, conf_matrix_SGK = SGKFoldAccuracy(X,Y,group,dummy)
            # end train-eval time
            end_time = time.time()
            #append accuracy and confusion matrix to the respective dicts
            accuracies['Impute by Group'].append(accuracy_TTS)
            accuracies['Impute by Group'].append(accuracy_SGK)
            conf_matricies['Impute by Group'].append(conf_matrix_TTS)
            conf_matricies['Impute by Group'].append(conf_matrix_SGK)
            
            # calculate and append time take
            time_taken = (end_time-start_time) 
            runtime['Impute by Group'].append(time_taken)
            
            
        elif sol == 'Impute by Feature':
            # Get cat and num features with winzorize flag as True(winzorized data)
            numeric_features,categorical_features,Y,group = LoadData('10kData.csv',True)
            # Impute by FEATURE
            numeric_features,categorical_features = ImputeByFeature(numeric_features,categorical_features)

            
            # OH-encode categorical feature and concat with num features
            X = OneHotEncoding(numeric_features,categorical_features)
            
            # start train-eval time                   
            start_time = time.time()
            # get accuracy and conf matrix for both eval methods
            accuracy_TTS, conf_matrix_TTS = TrainTestSplit(X,Y,dummy)
            accuracy_SGK, conf_matrix_SGK = SGKFoldAccuracy(X,Y,group,dummy)
            # end train-eval time
            end_time = time.time()
            #append accuracy and confusion matrix to the respective dicts
            accuracies['Impute by Feature'].append(accuracy_TTS)
            accuracies['Impute by Feature'].append(accuracy_SGK)
            conf_matricies['Impute by Feature'].append(conf_matrix_TTS)
            conf_matricies['Impute by Feature'].append(conf_matrix_SGK)
            
            # calculate and append time take
            time_taken = (end_time-start_time) 
            runtime['Impute by Feature'].append(time_taken)
            
    return accuracies,conf_matricies,runtime
            
def C2Results():
    solutions =  ['Impute by Group','Impute by Feature']
    accuracies,conf_matricies,runtime = MissingDataStrategy(solutions)
    print('Run Time:',runtime)
    print('Accuracy:',accuracies)
    print(conf_matricies)
   
    
if __name__ == "__main__":
    # get C1 results
    #C1Results(dummy=False)
    
    # get C2 results
    C2Results()
    

   

    