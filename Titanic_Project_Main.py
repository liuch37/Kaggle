#Machine lerning project: Classificaiton for Titanic survival 
#Command: python Titanic_Project_Main.py
import numpy
import random
import csv
from Data_Preprocess import Preprocess_v1, Preprocess_v2
from DNN import DNN_Model
from Random_Forest import RF_Model
from AdaBoost import AdaBoost_Model
from SVM import SVM_Model

#Fix random seed for reproducibility 
numpy.random.seed(3)
random.seed(6)

print("Hi, success for importing libraries!")

dataset, dataset_test = Preprocess_v2()

#Feature Randomization:
random.shuffle(dataset)

#Split dataset into train set (80%) and validation set (20%)
N_Total = len(dataset)
N_Train = int(N_Total * 0.8)
dataset_train = dataset[0:N_Train]
dataset_valid = dataset[N_Train+1:]

X_train = dataset_train[:,1:]
Y_train = dataset_train[:,0]

X_valid = dataset_valid[:,1:]
Y_valid = dataset_valid[:,0]

X_test = dataset_test[:,:]

#Train and test with different models:
#y = DNN_Model(X_train,Y_train,X_valid,Y_valid,X_test)
#y = RF_Model(X_train,Y_train,X_valid,Y_valid,X_test)
#y = AdaBoost_Model(X_train,Y_train,X_valid,Y_valid,X_test)
y = SVM_Model(X_train,Y_train,X_valid,Y_valid,X_test)

#Write output to output file:
y_convert = []
for item in y:
    y_convert.append(item.item())
myFile = open('Test_output.txt', 'w')
print("Writing predicted output in test set to the csv file!")
for item in y_convert:
    myFile.write("%d\n" % item)
myFile.close()




