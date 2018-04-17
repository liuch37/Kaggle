#K Nearest Neighbor Classifier
#ML framework: scikit-learn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def KNN_Model (X_train, Y_train, X_valid, Y_valid, X_test):
    #Model building for KNN
    #Hyperparameter List:
    neighbors = [5] #Number of neighbors, default = 5
    weights = 'distance' #'uniform', 'distance'
    algo = 'auto' #'ball_tree', 'kd_tree', 'brute'
    leave = [30] #Leave size, default = 30
    for n in neighbors:
        for l in leave:
            KNNC = KNeighborsClassifier(n_neighbors = n, weights = weights, algorithm = algo, leaf_size = l)
            print("Train with KNN model!")
            KNNC.fit(X_train,Y_train)
            #Classification performance:
            print("Hyperparameter: [",n,", ",l,"]->Training accuracy: ", accuracy_score(Y_train,KNNC.predict(X_train)))
            print("Hyperparameter: [",n,", ",l,"]->Validation accuracy: ", accuracy_score(Y_valid,KNNC.predict(X_valid)))

    #Prediction on the test set
    predictions = KNNC.predict(X_test)
    # round predictions
    y = [round(x) for x in predictions]

    return y

