
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def Graph(data):
    # Create histogram
    data.hist()
    # Save the histogram to this file name
    plt.savefig('histogram.png')

def Describe(data):
    data = data.head(10000)
    description = data.describe()
    # Separate the desired columns
    description = description.loc[['min', 'max', 'mean', '50%','count']]
    # Save the profile to this file name
    description.to_csv('profile.csv')

def PartitionDataset(data):
    data = data.head(10000)
    return data

def Str_toNumeric(data, feature_names):
    for feature_name in feature_names:
        data[feature_name] = pd.to_numeric(data[feature_name].str.replace('_', ''), errors='coerce')
   
    return data


if __name__ == "__main__":

    # missing values are represented by 99/-99 in the dataset
    data = pd.read_csv('train.csv')
    dataset = PartitionDataset(data)
    # TODO: Credit_Mix is str, _ should be na value, ,  Amount_invested_monthly null values are represented by 10000
    dataset = Str_toNumeric(dataset,['Age','Num_of_Loan','Num_of_Delayed_Payment','Annual_Income','Changed_Credit_Limit','Outstanding_Debt','Amount_invested_monthly'])
    dataset.loc[(dataset['Age'] < 18) | (dataset['Age'] > 100), 'Age'] = np.nan

    
    
    dataset.to_csv('10kData.csv')
    #Describe(data) # mean, median, max, min, num_of_missing values
    
    #Graph(data) # Create Histogram
    