# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 16:01:17 2018

@author: User
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(
    r'C:\eclipse-workspace\DeepLearningAndNN\data\Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_country = LabelEncoder()
X[:, 1] = labelencoder_country.fit_transform(X[:, 1])


labeleencoder_gender = LabelEncoder()
X[:, 2] = labeleencoder_gender.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()

# Removing first column to avoid dummy variable trap
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# MAKING OF THE ANN
import keras
from keras.models import Sequential  # Used to initialize our network
from keras.layers import Dense, Activation  # Used to create layers in our NN

# There are two ways to create a NN:
# 1. As a sequence of layers
# 2.  As a graph
# We will create our NN as a Sequence of layers
classifier_NN = Sequential()

# Adding the input layer and the first hidden layer
# We choose RELU function for the hidden layer
# AND sigmoid function for the output layer

# output - number of output nodes you want to add in this layer
# TIP: Choose number of nodes in the hidden layer
# as the average of number of nodes in the input + output layer
# number of nodes in the input layer is 11 - for the # of parameters
# number of nodes in the output layer is 1 - because we have a binary classifier
# output = (11+1)/2 = 6
classifier_NN.add(Dense(6, activation='relu', input_dim=11))

# Adding the second input layer
classifier_NN.add(Dense(6, activation='relu'))

# Adding the output layer
classifier_NN.add(Dense(1, activation='sigmoid'))

# Compiling the ANN - means applying stochastic Gradient descent to our NN
# optiizer='adam' - type of stochastic gradient descent algorithm
classifier_NN.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])

# Fitting the ANN to the training set
# batch_size and epochs are arbitrary, choosing the correct number of
# batches and epochs is an art
my_history = classifier_NN.fit(X_train, y_train,  batch_size=10, epochs=100)

def show_history(history):
    if (history):
        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        print("Train accuracy: ", history.history['acc'])
        print("Test accuracy: ", history.history['val_acc'])
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        print("Train loss: ", history.history['loss'])
        print("Test loss: ", history.history['val_loss'])
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    else:
        print("No history to show")

show_history(my_history)
# Predict the test set results
y_pred = classifier_NN.predict(X_test)
# Converting the values from probabilities to true/false values with given
# threshold
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



'''
# Predicting a single new observation
"""
Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000
"""
# data is provided in horizontal vector [[]] - like a 1 row
single_customer_prediction = classifier_NN.predict(np.array(
    sc.transform([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
single_customer_prediction = (single_customer_prediction > 0.5)


def plot_confusion_matrix(cm):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Paired)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(4)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

plot_confusion_matrix(cm)
'''