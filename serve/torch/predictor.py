from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import Any

import torch
import numpy as np
import numpy.typing as npt
from PIL import Image

import time
from torchvision import transforms, models
import torch
import numpy as np

class Predictor:
    def __init__(self, model_f: str, labels_f: str):
        with open(labels_f, "r") as fp:
            self.labels = json.load(fp)
        self.model = torch.load(model_f)
        self.model.eval()

    def _pre_process(self, img: str) -> npt.NDArray[Any]:
        img_b64dec = base64.b64decode(img)
        with Image.open(BytesIO(img_b64dec)) as fp:
            img_p = fp.convert("RGB")
        img_p = img_p.resize((224, 224))
        img = np.array(img_p)
        img = img.reshape(1, 3, 224, 224)
        return img

    def _infer(self, inp: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        net_in = torch.tensor(inp, dtype=torch.float32)
        with torch.inference_mode():
            net_out: torch.Tensor = self.model(net_in)
        net_out_np = net_out.numpy()
        return net_out_np

    def _post_process(self, net_out: npt.NDArray[np.float32]) -> dict[str, str]:
        predicted_idx = str(net_out.flatten().argmax(axis=0))
        result = self.labels[predicted_idx][1]
        return {"result": result}

    def __call__(self, inp: str) -> dict[str, str]:
        pre_process_res = self._pre_process(inp)
        infer_res = self._infer(pre_process_res)
        post_process_res = self._post_process(infer_res)
        return post_process_res

class MY_Predictor:
    def __init__(self):
        self.model_vgg = models.vgg16(pretrained=True)
        self.model_vgg.classifier = torch.nn.Linear(25088, 13)
        self.model_vgg.to('cpu')
        self.model_vgg.load_state_dict(torch.load('/home/ml_user/Downloads/vgg_weights_11.pth', map_location='cpu'))
        self.model_vgg.eval()

        self.test_data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.9123, std=0.2116)
        ])

    def __call__(self, inp: str) -> dict[str, str]:
        new_image = Image.fromarray(np.array(json.loads(inp), dtype='uint8'))
        img_t = self.test_data_transforms(new_image)
        batch_t = torch.unsqueeze(img_t, 0)
        result = torch.argmax(self.model_vgg(batch_t))
        # print(str(result.item()))
        # print(type(str(result.item())))
        return result

if __name__ == "__main__":
    # import time
    #
    # with open("../../client/img.jpeg", "rb") as fp:
    #     img = fp.read()
    #     img = base64.encodebytes(img).decode()
    #
    # predictor = Predictor("../../hub/model.pt", "imagenet.json")
    #
    # start = time.perf_counter()
    # res = predictor(img)
    # end = time.perf_counter()
    # print(f"Executed in {end-start} sec.")
    #
    # print(res)

    import time

    image = Image.open("../../client/52.jpg").convert("RGB")
    json_data = json.dumps(np.array(image).tolist())

    predictor = MY_Predictor()

    start = time.perf_counter()
    res = predictor(json_data)
    end = time.perf_counter()
    print(f"Executed in {end-start} sec.")

    print(res)


    # test_data_transforms = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=0.9123, std=0.2116)
    # ])

    # with open("../../client/52.jpg", "rb") as fp:
    #     img = fp.read()
    #     img = base64.encodebytes(img).decode()


    # start = time.perf_counter()
    # image = Image.open("../../client/52.jpg").convert("RGB")
    # json_data = json.dumps(np.array(image).tolist())
    # new_image = Image.fromarray(np.array(json.loads(json_data), dtype='uint8'))
    #
    # img_t = test_data_transforms(new_image)
    # batch_t = torch.unsqueeze(img_t, 0)

    # model_vgg = models.vgg16(pretrained=True)
    # model_vgg.classifier = torch.nn.Linear(25088, 13)
    # model_vgg.to('cpu')
    #
    # model_vgg.load_state_dict(torch.load('/home/ml_user/Downloads/vgg_weights_11.pth', map_location='cpu'))
    #
    # model_vgg.eval()
    #result = torch.argmax(model_vgg(batch_t))
    #print(result)


    #end = time.perf_counter()
    #print(f"Executed in {end-start} sec.")

    #print(result)
