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
import matplotlib.image

from Project_Spotify_502 import utils_spotify as u

if __name__ == "__main__":
    print ('entering run')
    dl_data = u.Data_DL()
    dl_data.save_images(path_X = dl_data.path_x_dl, format_ = 'png', size = 'small')
    dl_data.save_images(path_X = dl_data.path_x_dl, format_ = 'npy', size = 'small')