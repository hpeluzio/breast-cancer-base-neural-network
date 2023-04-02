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
classifier.add(Dense(units = 16, activation = 'relu', 
                     kernel_initializer = 'random_uniform'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

optimizer = keras.optimizers.Adam(learning_rate = 0.001, decay=0.0001, clipvalue=0.5)
classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

# classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
classifier.fit(training_predictors, training_class, batch_size=10, epochs=100)

weight0 = classifier.layers[0].get_weights()
print(weight0)
print(len(weight0)) # hidden layer and bias layer

weight1 = classifier.layers[1].get_weights()
print(weight1)
print(len(weight1)) # hidden layer and bias layer

weight2 = classifier.layers[2].get_weights() # last layer, 1 neuron
print(weight2)
print(len(weight2)) # hidden layer and bias layer

predictions = classifier.predict(test_predictors)
predictions = (predictions > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score

precision = accuracy_score(test_class, predictions)
matriz = confusion_matrix(test_class, predictions)

result = classifier.evaluate(test_predictors, test_class)

