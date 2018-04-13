#AdaBoost classifier
#ML framework: keras
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def AdaBoost_Model (X_train, Y_train, X_valid, Y_valid, X_test):
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
            print("Train with AdaBoost model!")
            AdaC.fit(X_train,Y_train)
            #Classification performance:
            print("Hyperparameter: [",N,", ",r,"]->Training accuracy: ", accuracy_score(Y_train,AdaC.predict(X_train)))
            print("Hyperparameter: [",N,", ",r,"]->Validation accuracy: ", accuracy_score(Y_valid,AdaC.predict(X_valid)))

    #Prediction on the test set
    predictions = AdaC.predict(X_test)
    # round predictions
    y = [round(x) for x in predictions]

    return y

