from flask import Flask, jsonify, Blueprint
from flask import request
import json
import numpy
import pandas as pd
from Predictor import Predictor

app = Flask(__name__)

@app.route("/predict", methods = ['POST'])
def GetPrediction(): 
    request_data = request.get_json()
    input_data = {
        "Name": [request_data["Name"]],
        "PatientAge": [request_data["PatientAge"]],
        "TimesPerDay": [request_data["TimesPerDay"]],
        "DiagnosticCode": [request_data["DiagnosticCode"]],
        "CitySize": [request_data["CitySize"]],
        "PillCost": [request_data["PillCost"]],
        "NumberOfProducts": [request_data["NumberOfProducts"]], 
        "KnownDoctorsVisits": [request_data["KnownDoctorsVisits"]],
        "Income": [request_data["Income"]],
        "DaysSinceLastViolation": [request_data["DaysSinceLastViolation"]],
        "Adhered": [request_data["Adhered"]]
    }
    input_data_df = pd.DataFrame(data=input_data)    
    predictor = Predictor()
    prediction = predictor.predict(input_data_df)
    return '{ "Adhered": ' + str(prediction[0][0]) + '}'

@app.route("/train")
def TrainModel():
    predictor = Predictor()
    predictor.train()
    return "Training Completed."

@app.route("/")
def HeartBeat():
    return "HeartBeat"
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("80"), debug=True)