FROM python:3.8.6-buster

COPY requirements.txt /requirements.txt

RUN apt-get update && apt-get install -y ffmpeg && apt-get install -y libsndfile1-dev && pip install -r requirements.txt

COPY saved_pipes/model_spotify.joblib /saved_pipes/model_spotify.joblib
COPY Project_Spotify_502 /Project_Spotify_502
COPY api /api

ENV SPOTIPY_CLIENT_ID="7d9bf533f1494fa0b2603b1f54cdadd0"
ENV SPOTIPY_CLIENT_SECRET="b2d8af2ad1da4f9caa3f5bf11baa98f7"

CMD uvicorn api.fast:app --reload --host 0.0.0.0 --port $PORT