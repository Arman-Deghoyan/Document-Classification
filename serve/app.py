from flask import Flask, request
from predictor import VGG_Predictor, FastTextVGGPredictor, FasttextPredictor

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict_func():

    payload_img = request.json["img"]
    selected_model = request.json["model"]

    if selected_model == "VGG":
        predictor = VGG_Predictor(
            "/app/Weights/VGG_Weights/vgg_weights_11.pth",
            "/app/VGG_Weights/VGG_Labels.json")

    elif selected_model == "VGGFasttext":

        predictor = FastTextVGGPredictor(
            "/app/Weights/VGGFasttext_Weights/Fasttext_VGG_17.pth",
            "/app/Weights/VGGFasttext_Weights/FastTextModelWeights.bin",
            "/app/VGGFasttext_Weights/VGG_Fasttext_Labels.json")

    elif selected_model == "Fasttext":

        predictor = FasttextPredictor(
            "/app/Weights/VGGFasttext_Weights/FastTextModelWeights.bin",
            "/app/VGGFasttext_Weights/VGG_Fasttext_Labels.json")

    else:
        raise Exception("Unknown model.")

    result = predictor(payload_img)

    return {"result": result}


if __name__ == "__main__":
    app.run()
