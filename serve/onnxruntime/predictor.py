from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import Any

import onnxruntime as ort
import numpy as np
import numpy.typing as npt
from PIL import Image


class Predictor:
    def __init__(self, model_f: str, labels_f: str):
        with open(labels_f, "r") as fp:
            self.labels = json.load(fp)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self.ort_session = ort.InferenceSession(
            model_f, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )

    def _pre_process(self, img: str) -> npt.NDArray[Any]:
        img_b64dec = base64.b64decode(img)
        with Image.open(BytesIO(img_b64dec)) as fp:
            img_p = fp.convert("RGB")
        img_p = img_p.resize((224, 224))
        img = np.array(img_p)
        img = img.reshape(1, 3, 224, 224)
        return img

    def _infer(self, inp: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        inp = inp.astype(np.float32)
        output = self.ort_session.run(None, {"input": inp})[0]
        return output

    def _post_process(self, net_out: npt.NDArray[np.float32]) -> dict[str, str]:
        predicted_idx = str(net_out.flatten().argmax(axis=0))
        result = self.labels[predicted_idx][1]
        return {"result": result}

    def __call__(self, inp: str) -> dict[str, str]:
        pre_process_res = self._pre_process(inp)
        infer_res = self._infer(pre_process_res)
        post_process_res = self._post_process(infer_res)
        return post_process_res


if __name__ == "__main__":
    import time

    with open("../../client/img.jpeg", "rb") as fp:
        img = fp.read()
        img = base64.encodebytes(img).decode()

    predictor = Predictor("../../hub/model.onnx", "imagenet.json")

    start = time.perf_counter()
    res = predictor(img)
    end = time.perf_counter()
    print(f"Executed in {end-start} sec.")

    print(res)
