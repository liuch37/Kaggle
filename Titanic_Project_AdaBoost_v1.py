#Machine learning project using AdaBoost Classifier for binary classification
#ML framework: scikit-learn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy
import random
import csv
#Fix random seed for reproducibility 
numpy.random.seed(396)
random.seed(87)

print("Hi, success for importing libraries!")

#Preprocess Data in train.csv and test.csv, which includes:
#(1) Feature Convertion - convert categorical variables into floating point variables
#(2) Feature Normalization - normalize feature to ~N(0,1)
#(3) Feature Completion - fill the gap for missing data
#(4) Randomization - randomize the features
#(5) Test data manipulation

#Skip passenger id and name
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
            #print(j,dataset_test[i][j])
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

#Feature Randomization:
random.shuffle(dataset)

#Split dataset into train set (80%) and validation set (20%)
N_Total = len(dataset)
N_Train = int(N_Total * 0.8)
dataset_train = dataset[0:N_Train]
dataset_valid = dataset[N_Train+1:]
#print(dataset_valid[0])

X_train = dataset_train[:,1:]
Y_train = dataset_train[:,0]

X_valid = dataset_valid[:,1:]
Y_valid = dataset_valid[:,0]

X_test = dataset_test[:,:]

#Model building for AdaBoost
#Hyperparameter List:
Num_Tree = 5
num_estimators = [1000]
rate = [0.5]
algo = 'SAMME.R'
for N in num_estimators:
    for r in rate:
        estimator = RandomForestClassifier(n_estimators = Num_Tree) #default=DecisionTreeClassifier
        AdaC = AdaBoostClassifier(base_estimator = estimator, n_estimators = N, learning_rate = r, algorithm = algo)
        AdaC.fit(X_train,Y_train)
        #Classification performance:
        print("Hyperparameter: [",N,", ",r,"]->Training accuracy: ", accuracy_score(Y_train,AdaC.predict(X_train)))
        print("Hyperparameter: [",N,", ",r,"]->Validation accuracy: ", accuracy_score(Y_valid,AdaC.predict(X_valid)))

#Prediction on the test set
predictions = AdaC.predict(X_test)
# round predictions
y = [round(x) for x in predictions]
#Write output to output file:
y_convert = []
for item in y:
    y_convert.append(item.item())
myFile = open('Test_output.txt', 'w')
for item in y_convert:
    myFile.write("%d\n" % item)
myFile.close()

print("Finiesed writing results in csv file!")



