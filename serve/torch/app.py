# from flask import Flask, request
# from predictor import Predictor
#
# predictor = Predictor("hub/model.pt", "imagenet.json")
#
# app = Flask(__name__)
#
#
# @app.route("/predict", methods=["POST"])
# def predict():
#     payload = request.json["img"]
#     result = predictor(payload)
#
#     return result

from torchvision import transforms, models
# import matplotlib.pyplot as plt
import torch
# load_model = torch.load('/content/drive/MyDrive/model_vgg_label_5_epoch_10.pth')


test_data_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.9123, std=0.2116)
            ])

from PIL import Image
from flask import Flask, request
# from predictor import Predictor

# predictor = Predictor("hub/model.pt", "imagenet.json")

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():

    img = Image.open('/home/ml_user/Downloads/final_project/train/cv/10062.jpg').convert('RGB')
    img_t = test_data_transforms(img)
    batch_t = torch.unsqueeze(img_t, 0)

    model_vgg = models.vgg16(pretrained=True)
    model_vgg.classifier = torch.nn.Linear(25088, 13)
    model_vgg.to('cpu')

    model_vgg.load_state_dict(torch.load('/home/ml_user/Downloads/vgg_weights_11.pth', map_location='cpu'))

    model_vgg.eval()
    result = torch.argmax(model_vgg(batch_t))
    print(result)

    return result