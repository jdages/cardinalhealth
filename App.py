from flask import Flask, jsonify, Blueprint
from flask import request
import json
import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

# data sources
training_data_url = "https://storage.googleapis.com/cardinalmldata/data.csv"
testing_data_url = "https://storage.googleapis.com/cardinalmldata/datatest.csv"

app = Flask(__name__)

@app.route("/predict")
def GetPrediction(date): 
    return "GetPrediction"          

@app.route("/train")
def TrainModel():
    return "TrainModel"

@app.route("/")
def HeartBeat():
    return "HeartBeat"

if __name__ == '__main__':
    app.run(debug=True)
