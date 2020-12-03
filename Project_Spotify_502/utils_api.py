import librosa
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from urllib.request import urlopen
import librosa
import numpy as np
import matplotlib.image

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

def generate_mp3_from_sample_url(track_id, url):
    sample_30s = urlopen(url)
    mp3_path = f"sample_{track_id}.mp3"
    output = open(mp3_path, 'wb')
    output.write(sample_30s.read())
    return mp3_path

def generate_spectrogram_url(track_id, url): 
    #loading with librosa
    mp3_path = generate_mp3_from_sample_url(track_id, url)
    x, sr = librosa.load(mp3_path, sr=44100, mono=True, duration = 29.976598639455784)
    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    mel = librosa.feature.melspectrogram(sr=sr, S=librosa.amplitude_to_db(stft))
    img = np.flip(mel, axis=0) # put low frequencies at the bottom in image
    img = 255-img # invert. make black==more energy
    image_path = f'{track_id}.png'
    matplotlib.image.imsave(image_path, img, cmap='gray')
    return None

if __name__ == '__main__': 
    song_urls = get_own_collection_preview_urls()
    print(song_urls)
