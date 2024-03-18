# README

## Author
Wadood Alam

## Date
18th March 2024

## Assignment
AI 539 Final Project: Credit Score Evaluation

## Dependencies / Imports Required

  - Python 
  - NumPy
  - Pandas
  - train_test_split
  - StratifiedGroupKFold
  - Scikit-learn
  - accuracy_score
  - confusion_matrix
  - HistGradientBoostingClassifier
  - DummyClassifier
  - scipy.stats.mstats: winsorize
  - time
  - imblearn.under_sampling: RandomUnderSampler
  - compute_sample_weight
  - re(regex)
  - matplotlib.pyplot 


## Instructions

### Program 1: Data Pre-processing & Data Profile

#### Execution 
1. Install the required dependencies using pip
2. Ensure Dataset(`train.csv`) is contained in the same directory
4. Run the program using the command `data_profile.py`
5. The program will print `Total number of missing values`, `Total number of missing values`, `Number of numerical features:`
6. The program will generate 4 csv files, 1 xlsx file, and 5 png files
7. `10kdata.csv`: The cleaned version of dataset containing 10,000 rows and 34 features
8. `Cat_profile.csv`: The data profile for the 4 relevant categorical features
9. `correlation_matrix.xlsx`: Correlation matrix
10. `profile.csv`: The data profile for numeric features
11. `Credit_Mix.png`: Representation of Credit_Mix feature
12. `Credit_Score.png`: Representation of Credit_Score feature
13. `missing_values.png`: Representation of the name of the feature with missing value and the number of values missing for each feature
14. `Month.png`: Representation of Month feature
15. `Payment_Behaviour.png`: Representation of Payment_Behaviour feature
    

### Program 2: Training and Evaluating

#### Execution 
1. Install the required dependencies using pip(if not installed previously)
2. Ensure Dataset(`10kdata.csv`) is contained in the same directory
4. Run the program using the command `train_eval.csv`
5. The program will output 3 dictionaries for accuracies, 3 dictionaries for runtime, 3 dictionaries for confusion matrices
6. The directories for confusion matrices and accuracies follow the following format: `{'Strategy name':[Train-test-split, Stratified Group-wise Cross-Validation],...}`
7. The runtime dictionary will follow the following format: `Run Time: {'Strategy name': [],...}`
8. The program will generate 1 csv file called `outliers.csv` to visualize outliers
 

## Files in the directory 
1. train.csv
2. 10kdata.csv
3. Cat_profile.csv
4. correlation_matrix.xlsx
5. profile.csv
6. Credit_Mix.png
7. Credit_Score.png
8. missing_values.png
9. Month.png
10. Payment_Behaviour.png
