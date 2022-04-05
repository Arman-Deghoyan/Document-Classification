from flask import Flask, request
from predictor import Predictor, MY_Predictor

#predictor = Predictor("hub/model.pt", "imagenet.json")
predictor = MY_Predictor()

app = Flask(__name__)


@app.route("/predict", methods=["GET", "POST"])
def predict_func():
    # print(request.json["img"])
    # payload = request.json["img"]
    # result = predictor(payload)

    image = Image.open("../../client/52.jpg").convert("RGB")
    json_data = json.dumps(np.array(image).tolist())
    print(json_data)
    result = predictor(json_data)

    return str(result.item())

#from torchvision import transforms, models
# import matplotlib.pyplot as plt
import torch
# load_model = torch.load('/content/drive/MyDrive/model_vgg_label_5_epoch_10.pth')


# test_data_transforms = transforms.Compose([
#                 transforms.Resize((224, 224)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=0.9123, std=0.2116)
#             ])

from PIL import Image
#from flask import Flask, request
import numpy as np
import json
# from predictor import Predictor

# predictor = Predictor("hub/model.pt", "imagenet.json")

#app = Flask(__name__)


# @app.route("/predict", methods=["POST"])
# def predict():

    # image = Image.open("../../client/52.jpg").convert("RGB")
    # json_data = json.dumps(np.array(image).tolist())
    # new_image = Image.fromarray(np.array(json.loads(json_data), dtype='uint8'))
    # img_t = test_data_transforms(new_image)
    # batch_t = torch.unsqueeze(img_t, 0)
    #
    # model_vgg = models.vgg16(pretrained=True)
    # model_vgg.classifier = torch.nn.Linear(25088, 13)
    # model_vgg.to('cpu')
    #
    # model_vgg.load_state_dict(torch.load('/home/ml_user/Downloads/vgg_weights_11.pth', map_location='cpu'))
    #
    # model_vgg.eval()
    # result = torch.argmax(model_vgg(batch_t))
    # print(result)

    # payload = request.json["img"]
    # result = predictor(payload)
    #
    # return result

if __name__ == "__main__":
    app.run()