FROM pytorch/pytorch:latest

ENV PYTHONPATH /app

RUN apt-get update
RUN apt-get install build-essential tesseract-ocr  -y

COPY ./requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

COPY . /app/

EXPOSE 5000

WORKDIR /app

ENTRYPOINT gunicorn app:app --bind 0.0.0.0:5000 --workers 1
