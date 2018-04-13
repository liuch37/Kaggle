#SVM Classifier
#ML framework: scikit-learn
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def SVM_Model (X_train, Y_train, X_valid, Y_valid, X_test):
    #Building Randomforest model:
    #Hyperparameter Lists:
    C = 2.1 #Penalty coefficient
    kernal = 'rbf' #Kernal function: 'linear', 'poly', 'rbf', 'sigmoid'
    gamma = [2.0] #Kernal coefficient for 'rbf', 'poly' and 'sigmoid'. Default='auto'

    for g in gamma:
        SVMC = SVC(C=C, kernel=kernal, gamma=g)
        print("Train with Support Vector Machine model!")
        SVMC.fit(X_train,Y_train)
        #Classification performance:
        print("Hyperparameter: [",g,"]->Training accuracy: ", accuracy_score(Y_train,SVMC.predict(X_train)))
        print("Hyperparameter: [",g,"]->Validation accuracy: ", accuracy_score(Y_valid,SVMC.predict(X_valid)))

    #Prediction on the test set
    predictions = SVMC.predict(X_test)
    # round predictions
    y = [round(x) for x in predictions]

    return y


