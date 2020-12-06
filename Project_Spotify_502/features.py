#!/usr/bin/env python3

# FMA: A Dataset For Music Analysis
# Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, Xavier Bresson, EPFL LTS2.

# All features are extracted using [librosa](https://github.com/librosa/librosa).
# Alternatives:
# * [Essentia](http://essentia.upf.edu) (C++ with Python bindings)
# * [MARSYAS](https://github.com/marsyas/marsyas) (C++ with Python bindings)
# * [RP extract](http://www.ifs.tuwien.ac.at/mir/downloads.html) (Matlab, Java, Python)
# * [jMIR jAudio](http://jmir.sourceforge.net) (Java)
# * [MIRtoolbox](https://www.jyu.fi/hum/laitokset/musiikki/en/research/coe/materials/mirtoolbox) (Matlab)

import os
import multiprocessing
import warnings
import numpy as np
from scipy import stats
import pandas as pd
import librosa
from tqdm import tqdm
import Project_Spotify_502.utils as utils
import Project_Spotify_502.utils_api as utils_api
from urllib.request import urlopen


def columns():
    feature_sizes = dict(chroma_stft=12, chroma_cqt=12, chroma_cens=12,
                         tonnetz=6, mfcc=20, rmse=1, zcr=1,
                         spectral_centroid=1, spectral_bandwidth=1,
                         spectral_contrast=7, spectral_rolloff=1)
    moments = ('mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max')

    columns = []
    for name, size in feature_sizes.items():
        for moment in moments:
            it = ((name, moment, '{:02d}'.format(i+1)) for i in range(size))
            columns.extend(it)

    names = ('feature', 'statistics', 'number')
    columns = pd.MultiIndex.from_tuples(columns, names=names)

    # More efficient to slice if indexes are sorted.
    return columns.sort_values()


def compute_features(tid):

    features = pd.Series(index=columns(), dtype=np.float32, name=tid)

    processed=0
    error=0
    error_id=None

    # Catch warnings as exceptions (audioread leaks file descriptors).
    warnings.filterwarnings('ignore', module='librosa')

    def feature_stats(name, values):
        features[name, 'mean'] = np.mean(values, axis=1)
        features[name, 'std'] = np.std(values, axis=1)
        features[name, 'skew'] = stats.skew(values, axis=1)
        features[name, 'kurtosis'] = stats.kurtosis(values, axis=1)
        features[name, 'median'] = np.median(values, axis=1)
        features[name, 'min'] = np.min(values, axis=1)
        features[name, 'max'] = np.max(values, axis=1)

    try:
        filepath = utils.get_audio_path(os.environ.get('AUDIO_DIR'), tid)
        x, sr = librosa.load(filepath, sr=None, mono=True)  # kaiser_fast
        f = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)
        feature_stats('zcr', f)

        cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
                                 n_bins=7*12, tuning=None))
        assert cqt.shape[0] == 7 * 12
        assert np.ceil(len(x)/512) <= cqt.shape[1] <= np.ceil(len(x)/512)+1

        f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats('chroma_cqt', f)
        f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats('chroma_cens', f)
        f = librosa.feature.tonnetz(chroma=f)
        feature_stats('tonnetz', f)

        del cqt
        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        assert stft.shape[0] == 1 + 2048 // 2
        assert np.ceil(len(x)/512) <= stft.shape[1] <= np.ceil(len(x)/512)+1
        del x

        f = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
        feature_stats('chroma_stft', f)

        f = librosa.feature.rms(S=stft)
        feature_stats('rmse', f)

        f = librosa.feature.spectral_centroid(S=stft)
        feature_stats('spectral_centroid', f)
        f = librosa.feature.spectral_bandwidth(S=stft)
        feature_stats('spectral_bandwidth', f)
        f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
        feature_stats('spectral_contrast', f)
        f = librosa.feature.spectral_rolloff(S=stft)
        feature_stats('spectral_rolloff', f)

        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
        del stft
        f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
        feature_stats('mfcc', f)
        processed+=1

    except Exception as e:
        print(f'error = {e}')
        error+=1
        error_id=tid

    return features, processed, error, error_id


def compute_features_from_mp3(tid):

    features = pd.Series(index=columns(), dtype=np.float32, name=tid)

    def feature_stats(name, values):
        features[name, 'mean'] = np.mean(values, axis=1)
        features[name, 'std'] = np.std(values, axis=1)
        features[name, 'skew'] = stats.skew(values, axis=1)
        features[name, 'kurtosis'] = stats.kurtosis(values, axis=1)
        features[name, 'median'] = np.median(values, axis=1)
        features[name, 'min'] = np.min(values, axis=1)
        features[name, 'max'] = np.max(values, axis=1)

    filepath = utils.get_audio_path(os.environ.get('AUDIO_DIR'), tid)
    x, sr = librosa.load(filepath, sr=None, mono=True)  # kaiser_fast
    f = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)
    feature_stats('zcr', f)

    cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
                                n_bins=7*12, tuning=None))
    assert cqt.shape[0] == 7 * 12
    assert np.ceil(len(x)/512) <= cqt.shape[1] <= np.ceil(len(x)/512)+1

    f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
    feature_stats('chroma_cqt', f)
    f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
    feature_stats('chroma_cens', f)
    f = librosa.feature.tonnetz(chroma=f)
    feature_stats('tonnetz', f)

    del cqt
    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    assert stft.shape[0] == 1 + 2048 // 2
    assert np.ceil(len(x)/512) <= stft.shape[1] <= np.ceil(len(x)/512)+1
    del x

    f = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
    feature_stats('chroma_stft', f)

    f = librosa.feature.rms(S=stft)
    feature_stats('rmse', f)

    f = librosa.feature.spectral_centroid(S=stft)
    feature_stats('spectral_centroid', f)
    f = librosa.feature.spectral_bandwidth(S=stft)
    feature_stats('spectral_bandwidth', f)
    f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
    feature_stats('spectral_contrast', f)
    f = librosa.feature.spectral_rolloff(S=stft)
    feature_stats('spectral_rolloff', f)

    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    del stft
    f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    feature_stats('mfcc', f)

    return features


def compute_features_from_filepath(name, filepath):

    features = pd.Series(index=columns(), dtype=np.float32, name=name)

    def feature_stats(name, values):
        features[name, 'mean'] = np.mean(values, axis=1)
        features[name, 'std'] = np.std(values, axis=1)
        features[name, 'skew'] = stats.skew(values, axis=1)
        features[name, 'kurtosis'] = stats.kurtosis(values, axis=1)
        features[name, 'median'] = np.median(values, axis=1)
        features[name, 'min'] = np.min(values, axis=1)
        features[name, 'max'] = np.max(values, axis=1)

    filepath = filepath
    x, sr = librosa.load(filepath, sr=None, mono=True)  # kaiser_fast
    f = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)
    feature_stats('zcr', f)

    cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
                                n_bins=7*12, tuning=None))
    assert cqt.shape[0] == 7 * 12
    assert np.ceil(len(x)/512) <= cqt.shape[1] <= np.ceil(len(x)/512)+1

    f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
    feature_stats('chroma_cqt', f)
    f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
    feature_stats('chroma_cens', f)
    f = librosa.feature.tonnetz(chroma=f)
    feature_stats('tonnetz', f)

    del cqt
    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    assert stft.shape[0] == 1 + 2048 // 2
    assert np.ceil(len(x)/512) <= stft.shape[1] <= np.ceil(len(x)/512)+1
    del x

    f = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
    feature_stats('chroma_stft', f)

    f = librosa.feature.rms(S=stft)
    feature_stats('rmse', f)

    f = librosa.feature.spectral_centroid(S=stft)
    feature_stats('spectral_centroid', f)
    f = librosa.feature.spectral_bandwidth(S=stft)
    feature_stats('spectral_bandwidth', f)
    f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
    feature_stats('spectral_contrast', f)
    f = librosa.feature.spectral_rolloff(S=stft)
    feature_stats('spectral_rolloff', f)

    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    del stft
    f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    feature_stats('mfcc', f)

    return features


def compute_features_from_url(song_url):

    tid = song_url[0]
    features = pd.Series(index=columns(), dtype=np.float32, name=tid)

    processed=0
    error=0
    error_id=None

    # Catch warnings as exceptions (audioread leaks file descriptors).
    warnings.filterwarnings('ignore', module='librosa')

    def feature_stats(name, values):
        features[name, 'mean'] = np.mean(values, axis=1)
        features[name, 'std'] = np.std(values, axis=1)
        features[name, 'skew'] = stats.skew(values, axis=1)
        features[name, 'kurtosis'] = stats.kurtosis(values, axis=1)
        features[name, 'median'] = np.median(values, axis=1)
        features[name, 'min'] = np.min(values, axis=1)
        features[name, 'max'] = np.max(values, axis=1)

    try:
        # generating temporary mp3 path
        sample_30s = urlopen(song_url[1])
        mp3_path = f"temp_{song_url[0]}.mp3"
        output = open(mp3_path, 'wb')
        output.write(sample_30s.read())

        #loading with librosa
        x, sr = librosa.load(mp3_path, sr=None, mono=True)

        #remove temp file (did this because it seemed to not remove it)
        try:
            os.remove(mp3_path)
        except Exception:
            print("all good bro, temp audio file already wiped")

        f = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)
        feature_stats('zcr', f)

        cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
                                 n_bins=7*12, tuning=None))
        assert cqt.shape[0] == 7 * 12
        assert np.ceil(len(x)/512) <= cqt.shape[1] <= np.ceil(len(x)/512)+1

        f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats('chroma_cqt', f)
        f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats('chroma_cens', f)
        f = librosa.feature.tonnetz(chroma=f)
        feature_stats('tonnetz', f)

        del cqt
        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        assert stft.shape[0] == 1 + 2048 // 2
        assert np.ceil(len(x)/512) <= stft.shape[1] <= np.ceil(len(x)/512)+1
        del x

        f = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
        feature_stats('chroma_stft', f)

        f = librosa.feature.rms(S=stft)
        feature_stats('rmse', f)

        f = librosa.feature.spectral_centroid(S=stft)
        feature_stats('spectral_centroid', f)
        f = librosa.feature.spectral_bandwidth(S=stft)
        feature_stats('spectral_bandwidth', f)
        f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
        feature_stats('spectral_contrast', f)
        f = librosa.feature.spectral_rolloff(S=stft)
        feature_stats('spectral_rolloff', f)

        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
        del stft
        f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
        feature_stats('mfcc', f)
        processed+=1

    except Exception as e:
        print(f'error = {e}')
        error+=1
        error_id=tid

    return features, processed, error, error_id


def main():
    tracks = utils.load('../raw_data/fma_metadata/tracks.csv')
    features = pd.DataFrame(index=tracks.index,
                            columns=columns(), dtype=np.float32)

    # More than usable CPUs to be CPU bound, not I/O bound. Beware memory.
    nb_workers = int(1.5 * len(os.sched_getaffinity(0)))

    tids = tracks[tracks.index >= 30000][tracks['set','subset']=='small'].index
    # to do the other blocks of data
    # tracks[np.logical_and(tracks.index >= 30000,tracks.index < 60000)][tracks['set','subset'] == 'small']
    
    #initializing counters of fails vs success
    error_list=[]
    processed_count=0
    error_count=0
    
    # creating iterable from the function, nb of processors and list for the function to loop over
    pool = multiprocessing.Pool(nb_workers)
    it = pool.imap_unordered(compute_features, tids)

    # iteration
    for row, processed, error, error_id in it:
        features.loc[row.name] = row
        processed_count+=processed
        error_count+=error
        if error_id:
            error_list.append(error_id)

    save(features)
    print(f'''
            ###########################
            processed: {processed_count}
            errors: {error_count}
            ###########################
            errors log below:
            ''')

    for error_id in error_list:
        print(f'            {error_id}')
    
    print('            ###########################')


def main_own_collection(nb_of_tracks=5, offset=0):
    tracks = utils.load('../raw_data/fma_metadata/tracks.csv')
    # generate extracts list
    song_urls = utils_api.get_own_collection_preview_urls(nb_of_tracks=nb_of_tracks, offset=offset)

    features = pd.DataFrame(index=[f"{x[0]}_test" for x in song_urls], columns=columns(), dtype=np.float32)

    #initializing counters of fails vs success
    error_list=[]
    processed_count=0
    error_count=0
    
    # More than usable CPUs to be CPU bound, not I/O bound. Beware memory.
    nb_workers = int(1.5 * len(os.sched_getaffinity(0)))
    
    # creating iterable from the function, nb of processors and list for the function to loop over
    pool = multiprocessing.Pool(nb_workers)
    it = pool.imap_unordered(compute_features_from_url, song_urls)

    # iteration
    for row, processed, error, error_id in it:
        features.loc[row.name] = row
        processed_count+=processed
        error_count+=error
        if error_id:
            error_list.append(error_id)

    save(features, f"features_{offset}", 10)
    print(f'''
            ###########################
            processed: {processed_count}
            errors: {error_count}
            ###########################
            errors log below:
            ''')

    for error_id in error_list:
        print(f'            {error_id}')
    
    print('            ###########################')


def main_one(tid):
    '''
    this function returns a dataframe with the features only for the few tids passed as argument
    for instance tid = 2 or 5 tid = any id in our audio samples data
    '''
    tracks = utils.load('../raw_data/fma_metadata/tracks.csv')
    features = pd.DataFrame(index=tracks.index,
                            columns=columns(), dtype=np.float32)

    features = compute_features_from_mp3(tid)

    # writing custom save function instead of using the one in this module
    features.sort_index(axis=1, inplace=True)
    features.to_csv('features_new.csv', float_format='%.{}e'.format(10))


def save(features, filename='features_new',ndigits=10):

    # Should be done already, just to be sure.
    features.sort_index(axis=0, inplace=True)
    features.sort_index(axis=1, inplace=True)

    features.to_csv(f'{filename}.csv', float_format='%.{}e'.format(ndigits))


def test(features, ndigits):

    indices = features[features.isnull().any(axis=1)].index
    if len(indices) > 0:
        print('Failed tracks: {}'.format(', '.join(str(i) for i in indices)))

    tmp = utils.load('features.csv')
    np.testing.assert_allclose(tmp.values, features.values, rtol=10**-ndigits)


if __name__ == "__main__":
    list_of_df=[]
    for k in range(0,10):
        offset = 0+50*k
        main_own_collection(nb_of_tracks=50,offset=offset)
        list_of_df.append(pd.read_csv(f'features_{offset}.csv', index_col=0, header = [0,1,2]))
    features_test = pd.concat([df for df in list_of_df]).dropna()
    features_test.to_csv('../raw_data/features_test.csv')