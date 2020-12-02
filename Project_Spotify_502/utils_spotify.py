import dotenv
import pydot
import requests
import numpy as np
import pandas as pd
import ctypes
import shutil
import multiprocessing
import multiprocessing.sharedctypes as sharedctypes
import os.path
import ast
import librosa

# path_x = '../raw_data/fma_medium'
# path_y = '../raw_data/fma_metadata/tracks.csv'

# Number of samples per 30s audio clip.
# TODO: fix dataset to be constant.
NB_AUDIO_SAMPLES = 1321967
SAMPLING_RATE = 44100

# Load the environment from the .env file.
dotenv.load_dotenv(dotenv.find_dotenv())


class Data:

    # getting the csv absolute path
    abs_path = __file__.replace('Project_Spotify_502/utils_spotify.py', '')
    path_x_dl = os.path.join('/',abs_path, 'raw_data/fma_medium')
    path_x_ml = os.path.join('/',abs_path, 'raw_data/fma_metadata/features.csv')
    path_y = os.path.join('/',abs_path, 'raw_data/fma_metadata/tracks.csv')

    def __init__(self): 
        return None

    def load(self, filepath):

        filename = os.path.basename(filepath)

        if 'features' in filename:
            return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

        if 'echonest' in filename:
            return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

        if 'genres' in filename:
            return pd.read_csv(filepath, index_col=0)

        if 'tracks' in filename:
            tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

            COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                    ('track', 'genres'), ('track', 'genres_all')]
            for column in COLUMNS:
                tracks[column] = tracks[column].map(ast.literal_eval)

            COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                    ('album', 'date_created'), ('album', 'date_released'),
                    ('artist', 'date_created'), ('artist', 'active_year_begin'),
                    ('artist', 'active_year_end')]
            for column in COLUMNS:
                tracks[column] = pd.to_datetime(tracks[column])

            SUBSETS = pd.api.types.CategoricalDtype(categories=['small', 'medium', 'large'], ordered=True)
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(SUBSETS)

            COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                    ('album', 'type'), ('album', 'information'),
                    ('artist', 'bio')]
            for column in COLUMNS:
                tracks[column] = tracks[column].astype('category')

            return tracks


    def get_audio_path(self, audio_dir, track_id):
        """
        Return the path to the mp3 given the directory where the audio is stored
        and the track ID.
        Examples
        --------
        >>> import utils
        >>> AUDIO_DIR = os.environ.get('AUDIO_DIR')
        >>> utils.get_audio_path(AUDIO_DIR, 2)
        '../data/fma_small/000/000002.mp3'
        """
        tid_str = '{:06d}'.format(track_id)
        return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')


    def generate_size(self, df, size = 'medium', subset = 'training'):
        '''Acceptable sizes = small, medium, large''' 
        return df[df['set', 'subset'] <= size]

    def generate_subset(self, df, subset = 'training'): 
        '''Generates dataset corresponding to the subset indicated
        Acceptable subsets = training, validation, test'''
        return df[df['set', 'split'] == subset]

    def generate_dataset(self, path = None, size = 'medium'): 
        '''Generates the whole dataset based on tracks.csv
        Acceptable sizes = small, medium, large'''

        # gets the tracks.csv path
        if path == None: 
            path = self.path_y

        # generates the dataset corresponding on the size 
        tracks_medium = self.generate_size(self.load(path), size = size)
        
        data_train = self.generate_subset(tracks_medium)
        data_val = self.generate_subset(tracks_medium, subset = 'validation')
        data_test = self.generate_subset(tracks_medium, subset = 'test')
        return data_train, data_val, data_test

    def generate_y(self, path = None, size = 'medium', nb_genres = 8): 
        '''Generates y (track_id and corresponding genre) based on dataset
        Acceptable sizes = small, medium, large'''

        # gets tracks.csv path
        if path == None:
            path = self.path_y

        # generates the dataset 
        data_train, data_val, data_test = self.generate_dataset(path, size = size)

        y_train = data_train[('track', 'genre_top')]

        # generates list of genres based on number of tracks
        genres = list(y_train.value_counts().head(nb_genres).index)

        # filters sub_datasets
        y_train = y_train[y_train.isin(genres)]
        y_val = data_val.loc[data_val[('track', 'genre_top')].isin(genres),('track', 'genre_top')]
        y_test = data_test.loc[data_test[('track', 'genre_top')].isin(genres),('track', 'genre_top')]
        
        return y_train, y_val, y_test


class Data_ML(Data):

    def __init__(self): 
        return None


class Data_DL(Data):

    def __init__(self): 
        return None

    def list_of_files(self, path):
        return [os.path.join(path, directory, file) for directory in os.listdir(path) for file in os.listdir(os.path.join(path, directory))]

    def generator_spectogram(self, filename): 
        x, sr = librosa.load(filename, sr=None, duration = 29.976598639455784, mono=True)
        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        mel = librosa.feature.melspectrogram(sr=sr, S=librosa.amplitude_to_db(stft))
        return mel, sr

    def generate_X(self, path = None):
        if path == None: 
            path = self.path_x_dl
        filenames = self.list_of_files(path) 
        spectrograms = []
        for filename in filenames: 
            mel, sr  = self.generator_spectogram(filename)
            spectrograms.append(mel)
        filenames = [int(filename[-10:-4].lstrip('0')) for filename in filenames]
        filenames = {track_id : index for index, track_id in enumerate(filenames)}
        return np.array(spectrograms), filenames

    def generate_X_y_subsets(self, path_X = None, path_y = None): 
        if path_X == None: 
            path_X = self.path_x_dl
        if path_y == None: 
            path_y = self.path_y
        X, filenames = self.generate_X(path_X)
        y_train, y_val, y_test = self.generate_y(path_y)
        index_train = [value for key, value in filenames.items() if key in list(y_train.index)]
        index_val = [value for key, value in filenames.items() if key in list(y_val.index)]
        index_test = [value for key, value in filenames.items() if key in list(y_test.index)]
        X_train = np.array([X[i, :, :] for i in index_train])
        X_val = np.array([X[i, :, :] for i in index_val])
        X_test = np.array([X[i, :, :] for i in index_test])
        return (X_train, X_val, X_test), (y_train, y_val, y_test)

    def save_X_y(self, path_X, path_y):
        X, y = self.generate_X_y_subsets(path_X, path_y)
        X_train, X_val, X_test = X
        y_train, y_val, y_test = y
        np.save('X_train.npy', X_train)
        np.save('y_train.npy', y_train)
        np.save('X_val.npy', X_val)
        np.save('y_val.npy', y_val)
        np.save('X_test.npy', X_test)
        np.save('y_test.npy', y_test)
        return None