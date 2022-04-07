from flask import Flask, request
from predictor import VGG_Predictor, FastTextVGGPredictor, FasttextPredictor

app = Flask(__name__)

VGG_MODEL = VGG_Predictor(
            "/app/Weights/VGG_Weights/vgg_weights_11.pth",
            "/app/Weights/VGG_Weights/VGG_Labels.json")


VGGFasttext_MODEL = FastTextVGGPredictor(
            "/app/Weights/VGGFasttext_Weights/Fasttext_VGG_17.pth",
            "/app/Weights/VGGFasttext_Weights/FastTextModelWeights.bin",
            "/app/Weights/VGGFasttext_Weights/VGG_Fasttext_Labels.json")

Fasttext_MODEL = FasttextPredictor(
            "/app/Weights/VGGFasttext_Weights/FastTextModelWeights.bin",
            "/app/Weights/VGGFasttext_Weights/VGG_Fasttext_Labels.json")


@app.route("/predict", methods=["POST"])
def predict_func():

    payload_img = request.json["img"]
    selected_model = request.json["model"]

    if selected_model == "VGG":
        result = VGG_MODEL(payload_img)

    elif selected_model == "VGGFasttext":
        result = VGGFasttext_MODEL(payload_img)

    elif selected_model == "Fasttext":
        result = Fasttext_MODEL(payload_img)

    else:
        raise Exception("Unknown model.")

    return {"result": result}


if __name__ == "__main__":
    app.run()
