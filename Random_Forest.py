#Random Forest Classifier
#ML framework: scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def RF_Model (X_train, Y_train, X_valid, Y_valid, X_test):
    #Building Randomforest model:
    #Hyperparameter Lists:
    Num_Tree = 80 #Number of trees
    criterion = 'gini' #Parsing criterion
    min_samples_leaf = 1 #min number of samples required to be at a leaf node
    max_leaf_nodes = 200 #max number of leaf nodes

    RFC = RandomForestClassifier(n_estimators = Num_Tree, criterion = criterion, min_samples_leaf = min_samples_leaf, max_leaf_nodes = max_leaf_nodes)
    print("Train with Random Forest model!")
    RFC.fit(X_train,Y_train)

    #Classification performance:
    print("Training accuracy: ", accuracy_score(Y_train,RFC.predict(X_train)))
    print("Validation accuracy: ", accuracy_score(Y_valid,RFC.predict(X_valid)))

    #Prediction on the test set
    predictions = RFC.predict(X_test)
    # round predictions
    y = [round(x) for x in predictions]

    return y


