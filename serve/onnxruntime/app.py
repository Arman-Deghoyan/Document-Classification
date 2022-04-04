from flask import Flask, request
from predictor import Predictor

predictor = Predictor("hub/model.onnx", "imagenet.json")

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.json["img"]
    result = predictor(payload)
    return result
