from flask import Flask, request
from predictor import Predictor

predictor = Predictor("hub/model.pt", "imagenet.json")

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.json["img"]
    result = predictor(payload)
    return result
