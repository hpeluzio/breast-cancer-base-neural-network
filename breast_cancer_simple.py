 import pandas as pd

predictors = pd.read_csv('inputs_breast.csv')
classes = pd.read_csv('outputs_breast.csv')

from sklearn.model_selection import train_test_split
training_predictors, test_predictors, training_class, test_class = train_test_split(predictors, classes, test_size=0.25)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(units = 16, activation = 'relu', 
                     kernel_initializer = 'random_uniform', input_dim = 30))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
classifier.fit(training_predictors, training_class, batch_size=10, epochs=100)

