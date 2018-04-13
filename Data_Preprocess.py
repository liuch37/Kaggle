#Data preprocessing functions with different feature engineering
import numpy

def Preprocess_v1 ():

    #Preprocess Data in train.csv and test.csv, which includes:
    #(1) Feature Convertion - convert categorical variables into floating point variables
    #(2) Feature Normalization - normalize feature to ~N(0,1)
    #(3) Feature Completion - fill the gap for missing data
    #(4) Randomization - randomize the features
    #(5) Test data manipulation

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
    #(a) Categorize age into 5 groups, missing data with a random group
    #(b) Categorize Fare into 5 groups, missing data with a random group
    #(c) Categorize name title into several groups, missing data with a random group
    #(2) Feature Normalization - normalize feature to ~N(0,1)
    #(3) Feature Completion - fill the gap for missing data
    #(4) Randomization - randomize the features
    #(5) Test data manipulation

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


