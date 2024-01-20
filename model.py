import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pickle
from sklearn.metrics import accuracy_score

df = pd.read_csv('Iris.csv')
target = df['Species']
print(pickle.format_version)
print(target)
seed = 7
train_data, test_data = train_test_split(df, test_size=0.3, 
random_state= 7)

# separate the independent and target variables from training data
train_x = train_data.drop(columns=['Species'],axis=1)
train_y = train_data['Species']

# separate the independent and target variables from test data
test_x = test_data.drop(columns=['Species'],axis=1)
test_y = test_data['Species']

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier()
dt_model.fit(train_x,train_y)

# make predictions on test data
dt_predictions_test = dt_model.predict(test_x)


# Make pickle file of our model
pickle.dump(dt_model, open("model.pkl", "wb"))