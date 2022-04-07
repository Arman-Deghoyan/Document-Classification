import base64
import json

import requests
import sys

with open(sys.argv[1], "rb") as fp:
    img = fp.read()
    img = base64.encodebytes(img).decode()

request = {"img": img, "model": "Fasttext"}

with open("request.json", "w") as file:
    json.dump(request, file) 
