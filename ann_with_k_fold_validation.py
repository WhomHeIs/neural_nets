# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 18:26:51 2018

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#importing Keras and friends
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential  # Used to initialize our network
from keras.layers import Dense # Used to create layers in our NN


dataset = pd.read_csv(r'Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

labelencoder_country = LabelEncoder()
X[:, 1] = labelencoder_country.fit_transform(X[:, 1])
labeleencoder_gender = LabelEncoder()
X[:, 2] = labeleencoder_gender.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()

# Removing first column to avoid dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Building our NN
def build_classifier():
    classifier_NN = Sequential()
    classifier_NN.add(Dense(6, activation='relu', input_dim=11))
    classifier_NN.add(Dense(6, activation='relu'))
    classifier_NN.add(Dense(1, activation='sigmoid'))
    classifier_NN.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    return classifier_NN

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
#applying k-fold-cross validation
#cv - number of folds to cross-validate - usually 10
# 10 is an optimal empirical value to provide low bias and appropriate varience
# n_jobs = The number of CPUs to use to do the computation. -1 means 'all CPUs' 
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)
# to apply the result of k-fold cross-validation we need to compute the mean of all accuracies
mean_score = accuracies.mean()
varience = accuracies.std()


