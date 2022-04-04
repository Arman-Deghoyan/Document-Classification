FROM python:3.8.10

WORKDIR /app/train

COPY . /app/train

RUN pip install -r requirements.txt

CMD [ "python", "./train.py" ]