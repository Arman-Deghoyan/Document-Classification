FROM sourcemind-lesson-serve-base

COPY . /app

RUN pip install -r requirements.txt

ENTRYPOINT gunicorn app:app --bind 0.0.0.0:8000 --workers 1