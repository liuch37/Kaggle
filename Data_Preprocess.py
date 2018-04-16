#Data preprocessing functions with different feature engineering
import numpy
import random
import pandas as pd
import re

random.seed(666)

def Preprocess_v1 ():

    #Preprocess Data in train.csv and test.csv, which includes:
    #(1) Feature Convertion - convert categorical variables into floating point variables
    #(2) Feature Normalization - normalize feature to ~N(0,1)
    #(3) Feature Completion - fill the gap for missing data
    #(4) Test data manipulation

    #Skip passenger id and ticket number
    Cols = [1,2,5,6,7,8,9,10,12] #[Survived, Pclass, Sex, Age, SibSp, Parch, Ticket, Fare, Embarked]
    Cols_test = [1,4,5,6,7,8,9,11] #        [Pclass, Sex, Age, SibSp, Parch, Ticket, Fare, Embarked]
    dataset = numpy.loadtxt("train.csv",dtype='str',delimiter=",",skiprows=1,usecols=Cols)
    dataset_test = numpy.loadtxt("test.csv",dtype='str',delimiter=",",skiprows=1,usecols=Cols_test)

    #Feature Convertion for train_validation data:
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            if j == 0:
                dataset[i][j] = int(dataset[i][j])
            elif j == 1:
                dataset[i][j] = int(dataset[i][j])
            elif j == 2:
                if dataset[i][j] == 'male':
                    dataset[i][j] = 0
                else:
                    dataset[i][j] = 1
            elif j == 3:
                if dataset[i][j] == '':
                    dataset[i][j] = -1
                else:
                    dataset[i][j] = float(dataset[i][j])
            elif j == 4:
                dataset[i][j] = int(dataset[i][j])
            elif j == 5:
                dataset[i][j] = int(dataset[i][j])
            elif j == 6:
                if dataset[i][j] != 'LINE':
                    dataset[i][j] = int(dataset[i][j].split(' ')[-1])
                else:
                    dataset[i][j] = -1
            elif j == 7:
                dataset[i][j] = float(dataset[i][j])
            elif j == 8:
                if dataset[i][j] == "S":
                    dataset[i][j] = 0
                elif dataset[i][j] == "C":
                    dataset[i][j] = 1
                else: #Q
                    dataset[i][j] = 2
    dataset = dataset.astype(numpy.float)
    
    #Feature Convertion for test data:
    for i in range(len(dataset_test)):
        for j in range(len(dataset_test[0])):
            if j == 0:
                dataset_test[i][j] = int(dataset_test[i][j])
            elif j == 1:
                if dataset_test[i][j] == 'male':
                    dataset_test[i][j] = 0
                else:
                    dataset_test[i][j] = 1
            elif j == 2:
                if dataset_test[i][j] == '':
                    dataset_test[i][j] = -1
                else:
                    dataset_test[i][j] = float(dataset_test[i][j])
            elif j == 3:
                dataset_test[i][j] = int(dataset_test[i][j])
            elif j == 4:
                dataset_test[i][j] = int(dataset_test[i][j])
            elif j == 5:
                if dataset_test[i][j] != 'LINE':
                    dataset_test[i][j] = int(dataset_test[i][j].split(' ')[-1])
                else:
                    dataset_test[i][j] = -1
            elif j == 6:
                if dataset_test[i][j] == '':
                    dataset_test[i][j] = -1
                else:
                    dataset_test[i][j] = float(dataset_test[i][j])
            elif j == 7:
                if dataset_test[i][j] == "S":
                    dataset_test[i][j] = 0
                elif dataset_test[i][j] == "C":
                    dataset_test[i][j] = 1
                else: #Q
                    dataset_test[i][j] = 2
    
    dataset_test = dataset_test.astype(numpy.float)

    print("Hi, finishing with data loading and data preprocessing!")
    
    #Feature Normalization:
    Mean = (numpy.mean(dataset,axis=0))
    Std = (numpy.std(dataset,axis=0))
    for j in range(1,len(dataset[0])):
        for i in range(len(dataset)):
            if dataset[i][j] == -1:
                dataset[i][j] = Mean[j]/Std[j]
            else:
                dataset[i][j] = (dataset[i][j] - Mean[j]) / Std[j]
    
    for j in range(len(dataset_test[0])):
        for i in range(len(dataset_test)):
            if dataset_test[i][j] == -1:
                dataset_test[i][j] = Mean[j+1]/Std[j+1]
            else:
                dataset_test[i][j] = (dataset_test[i][j] - Mean[j+1]) / Std[j+1]
    
    print("Hi, finishing feature normalization!")
        
    return dataset, dataset_test

def Preprocess_v2 ():

    #Preprocess Data in train.csv and test.csv, which includes:
    #(1) Feature Convertion - convert categorical variables into floating point variables and do segmentation
    #(a) Categorize age into groups by 5 years as duration, missing data with a random group
    #(b) Categorize Fare into groups by 20 bucks as duration, missing data with a random group
    #(c) Categorize name title into several groups, missing data with a random group
    #(2) Feature Normalization - normalize feature to ~N(0,1)
    #(3) Feature Completion - fill the gap for missing data    
    #(4) Test data manipulation

    #Skip passenger id and ticket number
    Cols = [1,2,4,5,6,7,8,10,12] #[Survived, Pclass, Name, Sex, Age, SibSp, Parch, Fare, Embarked]
    Cols_test = [1,3,4,5,6,7,9,11] #        [Pclass, Name, Sex, Age, SibSp, Parch, Fare, Embarked]
    dataset = numpy.loadtxt("train.csv",dtype='str',delimiter=",",skiprows=1,usecols=Cols)
    dataset_test = numpy.loadtxt("test.csv",dtype='str',delimiter=",",skiprows=1,usecols=Cols_test)    

    #Feature Convertion for train_validation data:
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            if j == 0:
                dataset[i][j] = int(dataset[i][j])
            elif j == 1:
                dataset[i][j] = int(dataset[i][j])
            elif j == 2:
                dataset[i][j] = dataset[i][j].split(' ')[1]
                if dataset[i][j] == 'Mr.':
                    dataset[i][j] = 0
                elif dataset[i][j] == 'Mrs.':
                    dataset[i][j] = 1
                elif dataset[i][j] == 'Miss.':
                    dataset[i][j] = 2
                elif dataset[i][j] == 'Master.':
                    dataset[i][j] = 3
                elif dataset[i][j] == 'Dr.':
                    dataset[i][j] = 4
                elif dataset[i][j] == 'Mrs.':
                    dataset[i][j] = 5
                else:
                    dataset[i][j] = 6
            elif j == 3:               
                if dataset[i][j] == 'male':
                    dataset[i][j] = 0
                else:
                    dataset[i][j] = 1
            elif j == 4:
                if dataset[i][j] == '':
                    dataset[i][j] = random.randint(0,20)
                else:
                    dataset[i][j] = int(float(dataset[i][j]) / 5)
            elif j == 5:
                dataset[i][j] = int(dataset[i][j])
            elif j == 6:
                dataset[i][j] = int(dataset[i][j])
            elif j == 7:
                dataset[i][j] = int(float(dataset[i][j]) / 20)
            elif j == 8:
                if dataset[i][j] == "S":
                    dataset[i][j] = 0
                elif dataset[i][j] == "C":
                    dataset[i][j] = 1
                else: #Q
                    dataset[i][j] = 2
    dataset = dataset.astype(numpy.float)
    print(dataset[0])    
    
    #Feature Convertion for test data:
    for i in range(len(dataset_test)):
        for j in range(len(dataset_test[0])):
            if j == 0:
                dataset_test[i][j] = int(dataset_test[i][j])
            elif j == 1:
                dataset_test[i][j] = dataset_test[i][j].split(' ')[1]
                if dataset_test[i][j] == 'Mr.':
                    dataset_test[i][j] = 0
                elif dataset_test[i][j] == 'Mrs.':
                    dataset_test[i][j] = 1
                elif dataset_test[i][j] == 'Miss.':
                    dataset_test[i][j] = 2
                elif dataset_test[i][j] == 'Master.':
                    dataset_test[i][j] = 3
                elif dataset_test[i][j] == 'Dr.':
                    dataset_test[i][j] = 4
                elif dataset_test[i][j] == 'Mrs.':
                    dataset_test[i][j] = 5
                else:
                    dataset_test[i][j] = 6
            elif j == 2:
                if dataset_test[i][j] == 'male':
                    dataset_test[i][j] = 0
                else:
                    dataset_test[i][j] = 1
            elif j == 3:
                if dataset_test[i][j] == '':
                    dataset_test[i][j] = random.randint(0,20)
                else:
                    dataset_test[i][j] = int(float(dataset_test[i][j]) / 5)
            elif j == 4:
                dataset_test[i][j] = int(dataset_test[i][j])
            elif j == 5:
                dataset_test[i][j] = int(dataset_test[i][j])
            elif j == 6:
                if dataset_test[i][j] == '':
                    dataset_test[i][j] = random.randint(0,10)
                else:
                    dataset_test[i][j] = int(float(dataset_test[i][j]) / 20)
            elif j == 7:
                if dataset_test[i][j] == "S":
                    dataset_test[i][j] = 0
                elif dataset_test[i][j] == "C":
                    dataset_test[i][j] = 1
                else: #Q
                    dataset_test[i][j] = 2
    
    dataset_test = dataset_test.astype(numpy.float)

    print("Hi, finishing with data loading and data preprocessing!")
    
    #Feature Normalization:
    Mean = (numpy.mean(dataset,axis=0))
    Std = (numpy.std(dataset,axis=0))
    #for j in range(1,len(dataset[0])):
    #    for i in range(len(dataset)):
            #dataset[i][j] = (dataset[i][j] - Mean[j]) / Std[j]
    
    #for j in range(len(dataset_test[0])):
    #    for i in range(len(dataset_test)):
            #dataset_test[i][j] = (dataset_test[i][j] - Mean[j+1]) / Std[j+1]
    
    print("Hi, finishing feature normalization!")
        
    return dataset, dataset_test

def Preprocess_v3 ():

    #Preprocess Data in train.csv and test.csv using pandas, which includes:
    #(1) Feature Convertion - convert categorical variables into floating point variables and do segmentation
    #(a) Categorize age into groups by 5 years as duration, missing data with a random group
    #(b) Categorize Fare into groups by 20 bucks as duration, missing data with a random group
    #(c) Categorize name title into several groups, missing data with another type
    #(2) Feature Completion - fill the gap for missing data    
    #(3) Test data manipulation
    dataset = pd.read_csv('./train.csv', header = 0)
    dataset_test  = pd.read_csv('./test.csv' , header = 0)
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    dataset_test['Sex'] = dataset_test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    # Getting histogram for Embarked
    EM = dataset['Embarked'].value_counts() #This is a series
    # Mapping Embarked by filling most common type to NA
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset_test['Embarked'] = dataset_test['Embarked'].fillna('S')
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    dataset_test['Embarked'] = dataset_test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    # Divide Age by 5 years as duration 
    M_Age = dataset['Age'].mean()    
    dataset['Age'] = dataset['Age'].fillna(M_Age)
    dataset_test['Age'] = dataset_test['Age'].fillna(M_Age)
    dataset['Age'] = (dataset['Age'] / 5).astype(int)
    dataset_test['Age'] = (dataset_test['Age'] / 5).astype(int)
    # Divide Fare by 20 as duration
    M_Fare = dataset['Fare'].mean()
    dataset['Fare'] = dataset['Fare'].fillna(M_Fare)
    dataset_test['Fare'] = dataset_test['Fare'].fillna(M_Fare)
    dataset['Fare'] = (dataset['Fare'] / 20).astype(int)
    dataset_test['Fare'] = (dataset_test['Fare'] / 20).astype(int)
    # Get name title
    for idx in range(len(dataset['Name'])):
        title_search = re.search(' ([A-Za-z]+)\.', dataset['Name'][idx])
        title_string = title_search.group(1)
        dataset['Name'][idx] = title_string
    # Map name:
    dataset['Name'] = dataset['Name'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Name'] = dataset['Name'].replace('Mlle', 'Miss')
    dataset['Name'] = dataset['Name'].replace('Ms', 'Miss')
    dataset['Name'] = dataset['Name'].replace('Mme', 'Mrs')
    dataset['Name'] = dataset['Name'].map( {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4} ).astype(int)
    dataset['Name'].fillna(5)

    for idx in range(len(dataset_test['Name'])):
        title_search = re.search(' ([A-Za-z]+)\.', dataset_test['Name'][idx])
        title_string = title_search.group(1)
        dataset_test['Name'][idx] = title_string
    # Map name:
    dataset_test['Name'] = dataset_test['Name'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset_test['Name'] = dataset_test['Name'].replace('Mlle', 'Miss')
    dataset_test['Name'] = dataset_test['Name'].replace('Ms', 'Miss')
    dataset_test['Name'] = dataset_test['Name'].replace('Mme', 'Mrs')
    dataset_test['Name'] = dataset_test['Name'].map( {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4} ).astype(int)
    dataset_test['Name'].fillna(5)

    # Drop features:
    drop_elements = ['PassengerId', 'Ticket', 'Cabin']
    dataset = dataset.drop(drop_elements, axis = 1)
    dataset_test = dataset_test.drop(drop_elements, axis = 1)

    dataset = dataset.values
    dataset_test = dataset_test.values

    print("Hi, finishing feature cleanup and preprocessing!")
    
    return dataset, dataset_test


