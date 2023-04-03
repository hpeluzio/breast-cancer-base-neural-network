import numpy as np
import pandas as pd
from keras.models import model_from_json

file = open('classifier_breast.json', 'r')
network_structure = file.read()
file.close()

classifier = model_from_json(network_structure)
classifier.load_weights('classifier_breast.h5')

new_register = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                  0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                  0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                  0.84, 158, 0.363]])

register_predict = classifier.predict(new_register)
register_predict_bin = (register_predict > 0.5)

# Evaluate the entire base, or even test database
predictors = pd.read_csv('inputs_breast.csv')
classes = pd.read_csv('outputs_breast.csv')

classifier.compile(loss='binary_crossentropy', optimizer='adam', 
                   metrics=['binary_accuracy'])
result = classifier.evaluate(predictors, classes) # We've got only 0.9209 accuracy
