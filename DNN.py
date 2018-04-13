#Deep Neural Network classifier
#ML framework: keras
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras import initializers

def DNN_Model (X_train, Y_train, X_valid, Y_valid, X_test):
    #Start to build deep learning model:
    #Hyperparameter List:
    Lambd = 0.001 #L2 regularization
    Num_Neuron = 10 #Number of neurons per layer
    Batch_size = 200 #Batch size
    Epochs = 20000 #Number of epochs

    #DNN Model: input layer with 8 features -> second layer with 10 neuron->tanh activation -> output layer with sigmoid activation function
    model = Sequential()
    model.add(Dense(Num_Neuron, input_dim=8, activation='tanh', kernel_regularizer=regularizers.l2(Lambd))) #kernel_initializer=initializers.he_normal(9487)
    model.add(Dense(1, activation='sigmoid'))
    #Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #Fit the model
    print("Train with DNN model!")
    model.fit(X_train, Y_train, epochs=Epochs, batch_size=Batch_size)
    #Evaluate the model
    scores_train = model.evaluate(X_train, Y_train)
    print("Training data Accuracy is:")
    print("%s: %.2f%%" % (model.metrics_names[1], scores_train[1]*100))
    #Evaluate the validation set
    scores_valid = model.evaluate(X_valid, Y_valid)
    print("Validation data Accuracy is:")
    print("%s: %.2f%%" % (model.metrics_names[1], scores_valid[1]*100))

    #Prediction on the test set
    predictions = model.predict(X_test)
    # round predictions
    y = [round(x[0]) for x in predictions]

    return y
