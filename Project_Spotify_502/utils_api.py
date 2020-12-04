import librosa
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth

def get_own_collection_preview_urls(nb_of_tracks=5, offset=0):
    '''
    Returns list with tids and preview urls
    '''
    scope = "user-library-read"

    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))
    results = sp.current_user_saved_tracks(limit=nb_of_tracks, offset=offset)

    song_urls = []
    for idx, item in enumerate(results['items']):
        tid = f'200{idx:03}'
        track = item['track']
        song_url = track['preview_url']
        song_urls.append((tid, song_url))
    
    return song_urls
