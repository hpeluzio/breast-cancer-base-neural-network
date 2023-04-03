import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

predictors = pd.read_csv('inputs_breast.csv')
classes = pd.read_csv('outputs_breast.csv')
