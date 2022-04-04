# from __future__ import annotations
#
# import base64
# import json
# from io import BytesIO
# from typing import Any
#
# import torch
# import numpy as np
# import numpy.typing as npt
# from PIL import Image
#
#
# class Predictor:
#     def __init__(self, model_f: str, labels_f: str):
#         with open(labels_f, "r") as fp:
#             self.labels = json.load(fp)
#         self.model = torch.load(model_f)
#         self.model.eval()
#
#     def _pre_process(self, img: str) -> npt.NDArray[Any]:
#         img_b64dec = base64.b64decode(img)
#         with Image.open(BytesIO(img_b64dec)) as fp:
#             img_p = fp.convert("RGB")
#         img_p = img_p.resize((224, 224))
#         img = np.array(img_p)
#         img = img.reshape(1, 3, 224, 224)
#         return img
#
#     def _infer(self, inp: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
#         net_in = torch.tensor(inp, dtype=torch.float32)
#         with torch.inference_mode():
#             net_out: torch.Tensor = self.model(net_in)
#         net_out_np = net_out.numpy()
#         return net_out_np
#
#     def _post_process(self, net_out: npt.NDArray[np.float32]) -> dict[str, str]:
#         predicted_idx = str(net_out.flatten().argmax(axis=0))
#         result = self.labels[predicted_idx][1]
#         return {"result": result}
#
#     def __call__(self, inp: str) -> dict[str, str]:
#         pre_process_res = self._pre_process(inp)
#         infer_res = self._infer(pre_process_res)
#         post_process_res = self._post_process(infer_res)
#         return post_process_res
#
#
# if __name__ == "__main__":
#     import time
#
#     with open("../../client/img.jpeg", "rb") as fp:
#         img = fp.read()
#         img = base64.encodebytes(img).decode()
#
#     predictor = Predictor("../../hub/model.pt", "imagenet.json")
#
#     start = time.perf_counter()
#     res = predictor(img)
#     end = time.perf_counter()
#     print(f"Executed in {end-start} sec.")
#
#     print(res)
