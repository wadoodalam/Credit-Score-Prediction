'''
Author: Wadood Alam
Date: 6th March 2024
Class: AI 539
Final Project: Credit Score Evaluation
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re,warnings

warnings.filterwarnings("ignore")

def HistForNumData(data):
    # Create histogram
    data.hist()
    # Save the histogram to this file name
    plt.savefig('histogram.png')
    
def PlotForCatData(data):
    feature_names = ['Credit_Score', 'Credit_Mix', 'Payment_Behaviour', 'Payment_of_Min_Amount']
    for feature_name in feature_names:
        plt.figure(figsize=(10, 6))
        data[feature_name].value_counts().plot(kind='barh', color='black')
        plt.title(feature_name)
        plt.xlabel('Count of Each Class')
        plt.ylabel('Class')
        plt.tight_layout()
        plt.savefig(f'{feature_name}.png')

def DescribeCatData(data):
    data_info = data[['Credit_Score', 'Credit_Mix', 'Payment_Behaviour']].astype('object').describe()
    data_info.to_csv('Cat_Profile.csv')

def ConvertCreditHistoryAge(age_str):
    # Check if string type(as per dataset convention), if not then return as is
    if isinstance(age_str, str):
        # Check if '_' after stripping the value(which is null)
        if age_str.strip('_') == '':
            # return NA for missing values
            return pd.NA  
        # Split value by whitespace/empty space
        age_split = age_str.split()
        # Extract years and months from the splitted value
        years = int(age_split[0])
        months = int(age_split[3])
        return years + months / 12
    else:
        return age_str

def Describe(data):
    description = data.describe()
    # Separate the desired columns
    description = description.loc[['min', 'max', 'mean', '50%','count']]
    # Save the profile to this file name
    description.to_csv('profile.csv')

def PartitionDataset(data):
    data = data.head(10000)
    return data

def RemoveUnderscore(data, feature_names,Sting=False):
    if Sting:
        for feature_name in feature_names:
            # Lamda func that return the original element x if _ found. If only _ then put Null value
            data[feature_name] = data[feature_name].apply(lambda x: pd.NA if x.strip('_') == '' else x)
    else:  
        for feature_name in feature_names:
            data[feature_name] = pd.to_numeric(data[feature_name].str.replace('_', ''), errors='coerce')
    
    return data

def IdentifyNull(data):
    #-100 for Num_of_Loan
    data['Num_of_Loan'] = data['Num_of_Loan'].replace(-100,pd.NA)
    # NM for Payment_of_Min_Amount
    data['Payment_of_Min_Amount'] = data['Payment_of_Min_Amount'].replace('NM',pd.NA)
    # !@9#%8 for Payment_Behaviour
    data['Payment_Behaviour'] = data['Payment_Behaviour'].replace('!@9#%8',pd.NA)
    return data
    
def LoanFeatureSplitting(data):
    # set for unqiue loan type
    loans = set()
    # Split by ',' if not null and add to the set
    for loan in data['Type_of_Loan']:
        if isinstance(loan, str):
            split = loan.split(', ')
            loans.update(split)
    # set for unique clean loan values
    clean_loans = set()
    # Remove 'and' from the above derived loan set
    for loan in loans:
        if 'and ' in loan:
            clean_loans.add(loan.replace('and ','')) 
        else: 
            # if 'and' not present then just add to the cleaned set as values are already cleaned
            clean_loans.add(loan)
    
    # Create new feature for each loan with binary representation 
    for loan_type in clean_loans:
        # If the extracted loan(s) are present in the value, create and update the loan(s) col with 0 or 1
        data[loan_type] = data['Type_of_Loan'].apply(lambda x: 1 if isinstance(x, str) and loan_type in x else 0)
    # Drop the original feature
    data.drop(columns=['Type_of_Loan'], inplace=True)
    return data

def ConvertCreditScoreClass(data):
    credit_score = data['Credit_Score']
    classes_map = {'Good': 1, 'Standard': 2, 'Poor': 3}
    
    num_classes = []
    for classes in credit_score:
        num_val = classes_map[classes]
        num_classes.append(num_val)
    
    data = data.drop(columns=['Credit_Score'])
    data['Credit_Score'] = num_classes
    return data

def ConvertPaymentMinAmountClass(data):
    amount = data['Payment_of_Min_Amount']
    class_map = {'No':0, 'Yes':1}
    
    num_classes = []
    for classes in amount:
        if isinstance(classes, str):
            num_val = class_map[classes]
        else:
            num_val = pd.NA
        num_classes.append(num_val)
    data = data.drop(columns=['Payment_of_Min_Amount'])
    data['Payment_of_Min_Amount'] = num_classes
    return data


if __name__ == "__main__":

    
    data = pd.read_csv('train.csv')
    # Drop unneeded columns
    data = data.drop(columns=['ID','Name','SSN'])
    # Partition dataset to 10000 rows
    dataset = PartitionDataset(data)
  
    # underscores are cleaned
    dataset = RemoveUnderscore(dataset,['Age','Num_of_Loan','Num_of_Delayed_Payment','Annual_Income','Changed_Credit_Limit','Outstanding_Debt','Amount_invested_monthly'])
    dataset = RemoveUnderscore(dataset,['Occupation','Credit_Mix'],Sting=True)
    # Age is converted to decimal for Credit History Age
    dataset['Credit_History_Age'] = dataset['Credit_History_Age'].apply(ConvertCreditHistoryAge)
    # Identify and replace null values
    dataset = IdentifyNull(dataset)
    # Split the Type_of_Loan feature in binary features for all types of loans
    dataset = LoanFeatureSplitting(dataset)
    
    dataset = ConvertCreditScoreClass(dataset)
    dataset = ConvertPaymentMinAmountClass(dataset)
    Describe(dataset)
    DescribeCatData(dataset)
    dataset.to_csv('10kData.csv')
    #PlotForCatData(dataset) # mean, median, max, min, num_of_missing values
    
    #Graph(data) # Create Histogram
    