import base64
import json

import requests
import sys

with open(sys.argv[1], "rb") as fp:
    img = fp.read()
    img = base64.encodebytes(img).decode()

request = {"img": img, "model": "Fasttext"}


res = requests.post("http://localhost:5000/predict", json=request)
print(res.json()['result'])
