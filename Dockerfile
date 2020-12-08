FROM python:3.8.6-buster

COPY requirements.txt /requirements.txt

RUN apt-get update && apt-get install -y libsndfile1-dev && pip install -r requirements.txt

COPY saved_pipes/model_spotify.joblib /saved_pipes/model_spotify.joblib
COPY Project_Spotify_502 /Project_Spotify_502
COPY api /api

CMD uvicorn api.fast:app --reload --host 0.0.0.0 --port $PORT