# write some code for the API here
from fastapi import FastAPI
import joblib
import pandas as pd 
import librosa 
import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
from urllib.request import urlopen
from Project_Spotify_502.features import compute_features_from_url
from Project_Spotify_502.utils_api import get_one_url
from google.cloud import storage
BUCKET_NAME = 'project_spotify_pellissier'
MODEL_NAME = 'project_spotify_502'

app = FastAPI()

model = joblib.load('model.joblib')
boolean = True

@app.get("/")
def index():
    return {'key':boolean}

# define a root `/` endpoint
@app.get("/predict_genre/{key}")
def predict_fare(key):
    # key = track_id spotify de la clé 
    url = get_one_url(key)

    # TO DO : appliquer une fonction qui sort les features 
    X_test = pd.Dataframe(compute_features_from_url(url)[0])

    y_pred = model.predict(X_test)
    return {'prediction':f'{y_pred[0]}'}


@app.get("/hello/{name}")
def hello(name):
    return {"key": f"Hello, {name}!"}