
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# TODO: Need to handle outliers(and then missing values)

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

def RemoveUnderscore(data, feature_names,Sting=False):
    if Sting:
        for feature_name in feature_names:
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
   
    #dataset.loc[(dataset['Age'] < 18) | (dataset['Age'] > 100), 'Age'] = np.nan

    
    
    dataset.to_csv('10kData.csv')
    Describe(dataset) # mean, median, max, min, num_of_missing values
    
    #Graph(data) # Create Histogram
    