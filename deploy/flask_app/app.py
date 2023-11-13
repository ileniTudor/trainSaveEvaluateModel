# check original example here
# https://www.section.io/engineering-education/deploying-machine-learning-models-using-flask/
import json
from math import expm1

import numpy as np
import requests
import joblib
import pandas as pd
from flask import Flask, jsonify, request, render_template, make_response
from tensorflow import keras
import os

os.environ["FLASK_DEBUG"] = "1"

app = Flask(__name__, template_folder="templates")

transformer = joblib.load("deploy/flask_app/model/data_transformer.joblib")
model = keras.models.load_model("deploy/flask_app/model/house_prediction_model.h5")


# Initialize the flask class and specify the templates directory
# Default route set as 'home'
@app.route('/')
def home():
    return render_template('home.html')  # Render home.html


@app.route("/predict", methods=["GET", "POST"])
def index():
    data = request.json

    input_arr = []
    for i in range(13):
        input_arr.append(float(data["a" + str(i)]))
    input_arr = np.array([input_arr])

    print("input_arr", input_arr)
    prediction = model.predict(transformer.transform(input_arr))
    predicted_score = expm1(prediction.flatten()[0])

    response = make_response(str(predicted_score), 200)
    response.mimetype = "text/plain"
    return response