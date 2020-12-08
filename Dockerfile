FROM python:3.8.6-buster

COPY saved_pipes/model_spotify.joblib /model.joblib
COPY Project_Spotify_502 /Project_Spotify_502
COPY api /api
COPY requirements.txt /requirements.txt

RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --reload --host 0.0.0.0 --port $PORT