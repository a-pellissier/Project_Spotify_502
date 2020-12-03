import dotenv
import pydot
import requests
import numpy as np
import pandas as pd
import ctypes
import shutil
import multiprocessing
import multiprocessing.sharedctypes as sharedctypes
import os
import ast
import librosa
import csv

# path_x = '../raw_data/fma_medium'
# path_y = '../raw_data/fma_metadata/tracks.csv'

# Number of samples per 30s audio clip.
# TODO: fix dataset to be constant.
NB_AUDIO_SAMPLES = 1321967
SAMPLING_RATE = 44100

# Load the environment from the .env file.
dotenv.load_dotenv(dotenv.find_dotenv())


class Data():

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


    def generate_size(self, df, size = 'medium'):
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

        data_train = self.generate_subset(tracks_medium, subset = 'training')
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

    def get_data_numeric(self, set_size, nb_genres):
        # Load complete datasets
        tracks = self.load(self.path_y)
        features = self.load(self.path_x_ml)

        # Select columns =  set_size, set_split & target
        tracks_b = tracks[[('set', 'subset'), ('set', 'split'), ('track', 'genre_top')]]
        tracks_b.columns = tracks_b.columns.droplevel(0)

        # Select relevant set sizing
        tracks_m = tracks_b[tracks_b.subset <= set_size]

        # get top_genres
        genres = tracks_m.genre_top.value_counts().head(nb_genres).index.to_list()
        tracks_cl = tracks_m[tracks_m.genre_top.isin(genres)]

        # get split indexes
        train_index = tracks_cl[tracks_cl['split'] == 'training'].index
        val_index = tracks_cl[tracks_cl['split'] == 'validation'].index
        test_index = tracks_cl[tracks_cl['split'] == 'test'].index

        # train/val/test split
        X_train = features.loc[train_index]
        X_val = features.loc[val_index]
        X_test = features.loc[test_index]

        y_train = tracks_cl.loc[train_index, 'genre_top']
        y_val = tracks_cl.loc[val_index, 'genre_top']
        y_test = tracks_cl.loc[test_index, 'genre_top']

        return (X_train, X_val, X_test), (y_train, y_val, y_test)


class Data_DL(Data):

    def __init__(self):
        return None

    def list_of_files(self, path, directory):
        return [os.path.join(path, directory, file) for file in os.listdir(os.path.join(path, directory))]

    def generator_spectogram(self, filename):
        try:
            x, sr = librosa.load(filename, sr=44100, duration = 29.976598639455784, mono=True)
        except:
            return None
        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        mel = librosa.feature.melspectrogram(sr=sr, S=librosa.amplitude_to_db(stft))
        return mel, sr

    def generate_X(self, directory, path = None):
        if path == None:
            path = self.path_x_dl
        filenames = self.list_of_files(path, directory)
        spectrograms = []
        file_prob = []
        for filename in filenames:
            temp = self.generator_spectogram(filename)
            if temp != None:    
                mel, sr  = temp
                spectrograms.append(mel)
            else:
                file_prob.append(filename)
                print(f'File: {filename} could not be loaded')
        for filename in file_prob:
            filenames.remove(filename)
        filenames = [int(filename[-10:-4].lstrip('0')) for filename in filenames]
        filenames = {track_id : index for index, track_id in enumerate(filenames)}
        return np.array(spectrograms), filenames

    def generate_X_y_subsets(self, directory, path_X = None, path_y = None):
        if path_X == None:
            path_X = self.path_x_dl
        if path_y == None:
            path_y = self.path_y
        X, filenames = self.generate_X(directory, path_X)
        y_train, y_val, y_test = self.generate_y(path_y)
        index_train = [value for key, value in filenames.items() if key in list(y_train.index)]
        index_val = [value for key, value in filenames.items() if key in list(y_val.index)]
        index_test = [value for key, value in filenames.items() if key in list(y_test.index)]
        X_train = np.array([X[i, :, :] for i in index_train])
        X_val = np.array([X[i, :, :] for i in index_val])
        X_test = np.array([X[i, :, :] for i in index_test])

        def format_y(y):
            index = [key for key, value in filenames.items() if key in list(y.index)]
            y = y[y.index.isin(index)]
            y = pd.DataFrame(y)
            y.reset_index(inplace = True)
            y.columns = [''.join(col) for col in y.columns.values]
            y.rename({'trackgenre_top' : 'genre'}, axis = 1, inplace = True)
            y['id'] = y['track_id'].map(filenames)
            y.set_index('id', inplace = True, drop = True)
            return y.sort_index()

        y_train = format_y(y_train)
        y_val = format_y(y_val)
        y_test = format_y(y_test)

        return (X_train, X_val, X_test), (y_train, y_val, y_test), filenames

    def save_X_y_dir(self, directory, save_path, path_X = None, path_y = None):
        import warnings
        warnings.filterwarnings("ignore")
        if path_X == None:
            path_X = self.path_x_dl
        if path_y == None:
            path_y = self.path_y

        X, y, filenames = self.generate_X_y_subsets(directory, path_X, path_y)
        X_train, X_val, X_test = X
        y_train, y_val, y_test = y
        np.save(os.path.join(save_path, directory, f'X_train_{directory}.npy'), X_train)
        np.save(os.path.join(save_path, directory, f'X_val_{directory}.npy'), X_val)
        np.save(os.path.join(save_path, directory, f'X_test_{directory}.npy'), X_test)

        y_val.to_csv(os.path.join(save_path, directory, f'y_val_{directory}.csv'))
        y_train.to_csv(os.path.join(save_path, directory, f'y_train_{directory}.csv'))
        y_test.to_csv(os.path.join(save_path, directory, f'y_test_{directory}.csv'))

        w = csv.writer(open(os.path.join(save_path, directory, f"filenames_{directory}.csv"), "w"))

        for key, val in filenames.items():
            w.writerow([key, val])
        return None
    
    def save_X_y(self, save_path, path_X = None, path_y = None):
        if path_X == None:
            path_X = self.path_x_dl
        if path_y == None:
            path_y = self.path_y
        directories = [os.path.join(path_X, directory)[-3:] for directory in os.listdir(path_X)]
        for directory in directories:
            save_directory = os.path.join(save_path, directory)
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            print(directory)
            print(f'++++Starting generation of spectrograms for {directory}++++')
            self.save_X_y_dir(directory, save_path, path_X, path_y)
            print(f'++++Successfully generated spectrograms for {directory}++++')
        return None