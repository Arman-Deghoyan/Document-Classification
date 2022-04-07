from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import Any

import numpy.typing as npt
from PIL import Image
from torchvision import transforms, models
import torch
import numpy as np
import time
from VGGFastTextModel import VGGFasttextModel
import pytesseract
from nltk.stem import SnowballStemmer
from text_utils import normalize, remove_punctuation, remove_stopwords_spacy, \
                       remove_non_eng_words, textLower
import fasttext

snowball = SnowballStemmer("english")


class VGG_Predictor:
    def __init__(self, model_file: str, labels_f: str):
        with open(labels_f, "r") as fp:
            self.labels = json.load(fp)
        self.model_vgg = models.vgg16(pretrained=False)
        self.model_vgg.classifier = torch.nn.Linear(25088, 13)
        self.model_vgg.to('cpu')
        self.model_vgg.load_state_dict(torch.load(model_file, map_location='cpu'))
        self.model_vgg.eval()

        self.test_data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.9123, std=0.2116)
        ])

    def _pre_process(self, img) -> npt.NDArray[Any]:
        img_b64dec = base64.b64decode(img)
        with Image.open(BytesIO(img_b64dec)) as fp:
            pil_image = fp.convert("RGB")
        img_t = self.test_data_transforms(pil_image)
        batch_t = torch.unsqueeze(img_t, 0)
        return batch_t

    def __call__(self, inp: str) -> str:
        pre_process_res = self._pre_process(inp)
        result = torch.argmax(self.model_vgg(pre_process_res))

        return self.labels[str(result.item())]


class FastTextVGGPredictor:
    def __init__(self, model_file: str, fasttext_model_file: str, labels_f: str):
        with open(labels_f, "r") as fp:
            self.labels = json.load(fp)
        self.model = VGGFasttextModel(13)
        self.model.to('cpu')
        self.model.load_state_dict(torch.load(model_file, map_location='cpu'))
        self.model.eval()

        self.ft_model = fasttext.load_model(fasttext_model_file)
        self.test_data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.9123, std=0.2116)
        ])

    def image_to_text(self, curr_img):

        ocr_df = pytesseract.image_to_data(curr_img, output_type='data.frame')
        ocr_df = ocr_df.dropna().reset_index(drop=True)
        float_cols = ocr_df.select_dtypes('float').columns
        ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
        ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
        try:
            words = ' '.join([word for word in ocr_df.text if str(word) != 'nan'])
        except:
            words = ''

        return words

    def preprocess_raw_text(self, curr_text):
        text = normalize(curr_text)
        text = remove_punctuation(text)
        text = textLower(text)
        text = snowball.stem(text)
        text = remove_stopwords_spacy(text)
        text = remove_non_eng_words(text)
        return text

    def get_vector_for_text(self, text):
        if len(text.split()) == 0:
            return np.zeros(100, dtype=np.float32)
        return np.average([self.ft_model.get_word_vector(x) for x in text.split()], axis=0)

    def _pre_process(self, img):
        img_b64dec = base64.b64decode(img)
        with Image.open(BytesIO(img_b64dec)) as fp:
            pil_image = fp.convert("RGB")
        img_t = self.test_data_transforms(pil_image)
        batch_t = torch.unsqueeze(img_t, 0)

        # build fasttext_features.
        fasttext_features = self.get_vector_for_text(self.preprocess_raw_text(self.image_to_text(pil_image)))

        return batch_t, torch.tensor(fasttext_features)

    def __call__(self, inp: str) -> str:
        preprocessed_image, fasttext_features = self._pre_process(inp)
        result = torch.argmax(self.model(preprocessed_image, fasttext_features))

        return self.labels[str(result.item())]


class FasttextPredictor:
    def __init__(self, fasttext_model_file: str, labels_f: str):
        with open(labels_f, "r") as fp:
            self.labels = json.load(fp)
        self.ft_model = fasttext.load_model(fasttext_model_file)

    def image_to_text(self, curr_img):

        ocr_df = pytesseract.image_to_data(curr_img, output_type='data.frame')
        ocr_df = ocr_df.dropna().reset_index(drop=True)
        float_cols = ocr_df.select_dtypes('float').columns
        ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
        ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
        try:
            words = ' '.join([word for word in ocr_df.text if str(word) != 'nan'])
        except:
            words = ''

        return words

    def preprocess_raw_text(self, curr_text):

        text = normalize(curr_text)
        text = remove_punctuation(text)
        text = textLower(text)
        text = snowball.stem(text)
        text = remove_stopwords_spacy(text)
        text = remove_non_eng_words(text)

        return text

    def _pre_process(self, img):
        img_b64dec = base64.b64decode(img)
        with Image.open(BytesIO(img_b64dec)) as fp:
            pil_image = fp.convert("RGB")

        # build fasttext_features.
        processed_text = self.preprocess_raw_text(self.image_to_text(pil_image))

        return processed_text

    def __call__(self, inp: str) -> str:
        processed_text = self._pre_process(inp)
        result = self.ft_model.predict(processed_text)
        return self.labels[result[0][0].split('_')[-1]]


if __name__ == "__main__":

    with open("../client/52.jpg", "rb") as fp:
        img = fp.read()
        img = base64.encodebytes(img).decode()

    predictor = VGG_Predictor("../Weights/VGG_Weights/vgg_weights_11.pth", "../../VGG_Weights/VGG_Labels.json")

    start = time.perf_counter()
    res = predictor(img)
    end = time.perf_counter()
    print(f"Executed in {end-start} sec.")

    print(res)
