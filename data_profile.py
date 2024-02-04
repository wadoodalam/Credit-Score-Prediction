
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# TODO: Need to handle outliers(and then missing values)

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
    data_info = data[['Credit_Score', 'Credit_Mix', 'Type_of_Loan', 'Payment_Behaviour', 'Payment_of_Min_Amount']].astype('object').describe()
    data_info.to_csv('Cat_Profile.csv')

def ConvertCreditHistoryAge(data):
    # Convert the age to decimal values
    return data


    
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


if __name__ == "__main__":

    
    data = pd.read_csv('train.csv')
    dataset = PartitionDataset(data)
  
    # underscores are cleaned
    dataset = RemoveUnderscore(dataset,['Age','Num_of_Loan','Num_of_Delayed_Payment','Annual_Income','Changed_Credit_Limit','Outstanding_Debt','Amount_invested_monthly'])
    dataset = RemoveUnderscore(dataset,['Occupation','Credit_Mix'],Sting=True)
    #dataset = ConvertCreditHistoryAge(dataset)
    #dataset.loc[(dataset['Age'] < 18) | (dataset['Age'] > 100), 'Age'] = np.nan

    
    
    dataset.to_csv('10kData.csv')
    PlotForCatData(dataset) # mean, median, max, min, num_of_missing values
    
    #Graph(data) # Create Histogram
    