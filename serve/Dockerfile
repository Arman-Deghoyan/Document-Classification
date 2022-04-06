FROM pytorch/pytorch:latest

COPY . /app

ENV PYTHONPATH /app

RUN apt-get update
RUN apt-get install build-essential -y
RUN pip install -r /app/requirements.txt

EXPOSE 5000

ENTRYPOINT gunicorn app:app --bind 0.0.0.0:5000 --workers 1
