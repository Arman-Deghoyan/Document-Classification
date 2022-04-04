import base64
import requests


with open("img.jpeg", "rb") as fp:
    img = fp.read()
    img = base64.encodebytes(img).decode()

request = {"img": img}

res = requests.post("http://localhost:8001/predict", json=request)
print(res.json())
