from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pickle
from sklearn.tree import DecisionTreeClassifier

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    df = pd.read_csv('Iris.csv')
    target = df['Species']
    seed = 7
    train_data, test_data = train_test_split(df, test_size=0.3,random_state= 7)
    # separate the independent and target variables from training data
    train_x = train_data.drop(columns=['Species'],axis=1)
    train_y = train_data['Species']

    # separate the independent and target variables from test data
    test_x = test_data.drop(columns=['Species'],axis=1)
    test_y = test_data['Species']
    dt_model = DecisionTreeClassifier()
    dt_model.fit(train_x,train_y)
    # make predictions on test data
    dt_predictions_test = dt_model.predict(test_x)
    accuracy = accuracy_score(test_y,dt_predictions_test) * 100

    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    arr = np.array([[data1, data2, data3, data4]])
    pred = model.predict(arr)[0]
    return render_template('after.html', data=pred,accuracy=accuracy)


if __name__ == "__main__":
    app.run(debug=True)