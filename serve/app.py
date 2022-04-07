from flask import Flask, request
from predictor import VGG_Predictor, FastTextVGGPredictor, FasttextPredictor

app = Flask(__name__)

app.VGG_MODEL = None 
app.VGGFasttext_MODEL = None
app.Fasttext_MODEL = None 

@app.route("/predict", methods=["POST"])
def predict_func():

    payload_img = request.json["img"]
    selected_model = request.json["model"]

    if selected_model == "VGG":
        if app.VGG_MODEL is None:
            app.VGG_MODEL = VGG_Predictor(
                "/app/Weights/VGG_Weights/vgg_weights_11.pth",
                "/app/Weights/VGG_Weights/VGG_Labels.json")

        result = app.VGG_MODEL(payload_img)

    elif selected_model == "VGGFasttext":
        if app.VGGFasttext_MODEL is None:
            app.VGGFasttext_MODEL = FastTextVGGPredictor(
                "/app/Weights/VGGFasttext_Weights/Fasttext_VGG_17.pth",
                "/app/Weights/VGGFasttext_Weights/FastTextModelWeights.bin",
                "/app/Weights/VGGFasttext_Weights/VGG_Fasttext_Labels.json")

        result = app.VGGFasttext_MODEL(payload_img)

    elif selected_model == "Fasttext":
        if app.Fasttext_MODEL is None:
            app.Fasttext_MODEL = FasttextPredictor(
                "/app/Weights/VGGFasttext_Weights/FastTextModelWeights.bin",
                "/app/Weights/VGGFasttext_Weights/VGG_Fasttext_Labels.json")

        result = app.Fasttext_MODEL(payload_img)
    else:
        raise Exception("Unknown model.")

    return {"result": result}


if __name__ == "__main__":
    app.run()
