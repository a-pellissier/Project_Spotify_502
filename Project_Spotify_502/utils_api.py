import librosa
import os
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
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
        track = item['track']
        tid = track['id']
        # tid = f'200{idx:03}'
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

def get_own_collection_genres(nb_of_tracks=5, offset=0):
    '''
    Returns list with tids and genres associated with the tracks author
    '''
    # authorizing the scope 'user-library-read'
    scope = "user-library-read"
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))
    results = sp.current_user_saved_tracks(limit=nb_of_tracks, offset=offset)
    
    # looping on playlist tracks to get their artist and associated genres
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
    tids_list=[]
    genres_lists = []
    for idx, item in enumerate(results['items']):
        try: 
            track_id = item['track']['id']
            tid = track_id
            
            #get artist id of a given track
            lz_uri = f'spotify:track:{track_id}'
            results = spotify.track(lz_uri)
            artist_id = results['album']['artists'][0]['id']

            #get genres from the album
            lz_uri = f'spotify:artist:{artist_id}'
            spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
            results = spotify.artist(lz_uri)
            if results['genres'] != None:
                if tid != None:
                    genres_lists.append(results['genres'])
                    tids_list.append(tid)
        except Exception as e:
            print("skipped, error was: ", e)

    return (tids_list,genres_lists)

def genre_spotify_to_FMA(spotify_genre_list):
    '''
    give the function a spotify genre list (ie several genre tags for one given artist)
    it will convert it to one of the main 8 FMA genres or say it does not belong to any of those 8
    '''
    # listing genres and possible aliases
    eight_genres = {'Electronic':{'Electronic':0, 'house':0, 'techno':0},
                    'Experimental':{'Experimental':0}, 
                    'Folk':{'Folk':0, 'melancholia':0, }, 
                    'Hip-Hop':{'Hip hop':0, 'hip-hop':0, 'urbaine':0}, 
                    'Instrumental':{'Instrumental':0, 'acoustic':0, 'ambient':0,'piano':0}, 
                    'International':{'International':0,'brazil':0}, 
                    'Pop':{'Pop':0, 'indie-pop':0, 'dance':0}, 
                    'Rock':{'Rock':0, 'metal':0, 'psychedelic':0}}
    
    # take each item of the list and check if it contains any genre alias in our top8
    for item in spotify_genre_list:
        for genre in eight_genres.keys():
            for alias in eight_genres[genre].keys():
                if alias.lower() in item.lower():
                    eight_genres[genre][alias] += 1
    
    # classify on the basis of the most represented genre 
    eight_genres_summary={}
    for genre in eight_genres.keys():
        eight_genres_summary[genre]=0
        for alias in eight_genres[genre].keys():
            eight_genres_summary[genre] += eight_genres[genre][alias]
            
    #defining main genre as max occurences of aliases        
    import operator
    main_genre = max(eight_genres_summary.items(), key=operator.itemgetter(1))[0]
    
    return main_genre, eight_genres_summary

def gen_y_from_saved_collection(nb_of_iter=10):
    genres_lists = []
    tids_lists = []
    for k in range(0,nb_of_iter):
        print(k)
        offset = 0+50*k
        tids_lists.append(get_own_collection_genres(nb_of_tracks=50,offset=offset)[0])
        genres_lists.append(get_own_collection_genres(nb_of_tracks=50,offset=offset)[1])
    
    #making genres and tids flat instead of "block lists"
    flat_tids = []
    for sublist in tids_lists:
        print("tid lenght",len(sublist))
        for item in sublist:
            flat_tids.append(item)

    flat_genres = []
    for sublist in genres_lists:
        print("genres_length", len(sublist))
        for item in sublist:
            flat_genres.append(item)

    df = pd.DataFrame(dict(tid=flat_tids, genres=flat_genres))
    df['main_genre']=df.copy()['genres'].apply(lambda x: genre_spotify_to_FMA(x)[0])
    df['genre_summary']=df.copy()['genres'].apply(lambda x: genre_spotify_to_FMA(x)[1])
    df.set_index(keys='tid',inplace=True,drop=True)
    df.to_csv('genres.csv')
    return df


if __name__ == '__main__':
    gen_y_from_saved_collection(10)

    #ERROR BECAUSE TIDS AND GENRES DO NOT MATCHUP DUE TO SKIPS, SOLUTION IS TO NOT SAVE ONE OR THE OTHER WHEN THERE IS AN ERROR