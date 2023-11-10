# check original example here
# https://www.section.io/engineering-education/deploying-machine-learning-models-using-flask/
import json
from math import expm1

import numpy as np
import requests
import joblib
import pandas as pd
from flask import Flask, jsonify, request, render_template
from tensorflow import keras
import os

os.environ["FLASK_DEBUG"] = "1"

app = Flask(__name__, template_folder="templates")

transformer = joblib.load("model/data_transformer.joblib")


# Initialize the flask class and specify the templates directory
# Default route set as 'home'
@app.route('/home')
def home():
    return render_template('home.html')  # Render home.html


@app.route("/predict", methods=["GET", "POST"])
def index():
    data = request.json
    print("data", data)
    input_arr = []
    for i in range(13):
        input_arr.append(float(data["a" + str(i)]))
    print("input_arr", input_arr)
    input_arr = np.array([input_arr])
    # df = pd.DataFrame(input_arr, index=[0])
    model = keras.models.load_model("model/house_prediction_model.h5")
    prediction = model.predict(transformer.transform(input_arr))
    print("after transform", input_arr)
    predicted_score = expm1(prediction.flatten()[0])
    print("predicted_score", predicted_score)
    # resp = requests.post("predict", data=jsonify({"price": str(predicted_price)}),
    #                      headers={"Content-Type": "application/json"}, timeout=1.0)
    # return resp #jsonify({"price": str(predicted_price)})

    BASE = "http://127.0.0.1:5000/"
    payload = {"price": str(predicted_score)}
    headers = {"Content-Type": "application/json; charset=utf-8"}
    response = requests.post("/predict", json=payload, headers=headers)
    print("payload", payload)
    print("headers", headers)
    print("response.json()", response.json())

    return response.json()
