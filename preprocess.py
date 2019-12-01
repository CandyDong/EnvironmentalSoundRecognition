import pandas as pd
import numpy as np
import sys
import os

import matplotlib.pyplot as plt

#for loading and visualizing audio files
import librosa
import librosa.display

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# librosa for audio feature extraction
import librosa
import librosa.display
import gc
import pickle
import random
from multiprocessing import Pool
from PIL import Image

# plotly libraries
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly_express as px
import plotly.io as pio


pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
pd.options.display.max_rows = 5000

CLASS_ID = {0: "air_conditioner",
            1: "car_horn",
            2:"children_playing",
            3:"dog_bark",
            4:"drilling",
            5:"engine_idling",
            6:"gun_shot",
            7:"jackhammer",
            8:"siren",
            9:"street_music"}

def _get_meta_info(filename):
    infos = filename.split('-')
    return [int(info) for info in infos]

def input_to_target(opts):
    # audio files and their corresponding labels
    train_paths = [opts.data_path + "fold1/*.wav", opts.data_path + "fold2/*.wav"]
    # train_paths = [opts.data_path + "fold1/*.wav"]
    train_label_path = opts.data_path +  "train_labels.csv"
    test_paths =  [opts.data_path + "fold2/*.wav"]

    # input
    train_files, test_files = [], []
    for train_path in train_paths:
        train_files += glob.glob(train_path)
    for test_path in test_paths:
        test_files += glob.glob(test_path)

    train_labels, class_names = [], []
    for train_file in train_files:
        _, class_id, _, _ = _get_meta_info(train_file.split('/')[-1].strip('.wav'))
        # print("train_file={:s}, class_id={:d}".format(train_file, class_id))
        train_labels.append(int(class_id))
        class_names.append(CLASS_ID[int(class_id)])
    # csv storing information for training dataset
    train_file_df = pd.DataFrame({'file_paths': train_files, 
                                'labels': train_labels,
                                'class_names': class_names})

    test_labels, class_names = [], []
    for test_file in test_files:
        _, class_id, _, _ = _get_meta_info(test_file.split('/')[-1].strip('.wav'))
        test_labels.append(int(class_id))
        class_names.append(CLASS_ID[int(class_id)])
    # csv storing information for training dataset
    test_file_df = pd.DataFrame({'file_paths': test_files, 
                                'labels': test_labels,
                                'class_names': class_names})
    
    return train_file_df, test_file_df

def _audio_normalization(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+0.0001)
    return data-0.5

def load_audio_file(file_path, input_length=64000):
    data, sr = librosa.core.load(file_path, sr=16000) 
    # randomly crop desired length of data from the original audio file 
    if len(data) > input_length:
        max_offset = len(data)-input_length
        offset = np.random.randint(max_offset)
        data = data[offset:(input_length+offset)]    
    else:
        if input_length > len(data):
            max_offset = input_length - len(data)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
        data = _audio_normalization(data)
    return data

def load_audio_spectrogram(file_path, opts):
    wave, sr = librosa.load(file_path, sr=opts.sr)
    S = librosa.core.stft(y=wave)
    S_db = librosa.power_to_db(np.abs(S)**2,ref=np.max)
    return S_db

def load_audio_feature(file_path, opts):
    y, sr = librosa.load(file_path, sr=opts.sr)
    if y.ndim > 1:
        y = y[:, 0]
    y = y.T

    # Get features
    stft = np.abs(librosa.stft(y))
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=opts.sr, n_mfcc=40).T, axis=0)  # 40 values
    # zcr = np.mean(librosa.feature.zero_crossing_rate)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=opts.sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y, sr=opts.sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=opts.sr).T, axis=0)
    # tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=opts.sr).T, # tonal centroid features
    #                 axis=0)

    # Return computed features
    return mfccs, chroma, mel, contrast

def plot_mel_spectrogram(df, opts):
    class_set = set()
    output_dir = opts.plot_path + 'mel_spectrogram/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_path, class_name, class_id in zip(df['file_paths'], df['class_names'], df['labels']):
        if len(class_set) == 10:
            break
        if class_id in class_set:
            continue
        class_set.add(class_id)

        wave, sr = librosa.load(file_path, sr=opts.sr)
        mel_spec = librosa.feature.melspectrogram(y=wave, sr=opts.sr, n_mels=320, fmax=16000)
        
        plt.rcParams.update({'font.size': 13})
        plt.figure(figsize=(15, 6))
        librosa.display.specshow(librosa.power_to_db(mel_spec,ref=np.max), y_axis = 'mel', x_axis= 'time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('{:s}({:s})'.format(file_path, class_name))
        plt.savefig(output_dir+ class_name + '.png', dpi =300)
        plt.show()

def plot_time_amplitude(df, opts):
    class_names = df['class_names'].tolist()
    li = []
    class_set = set()
    output_dir = opts.plot_path + 'time_amplitude/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_paths = []
    class_names = []
    for file_path, class_name, class_id in zip(df['file_paths'], df['class_names'], df['labels']):
        if len(class_set) == 10:
            break
        if class_id in class_set:
            continue
        class_set.add(class_id)
        file_paths.append(file_path)
        class_names.append(class_name)

        wave = load_audio_file(file_path, input_length=opts.sr*opts.audio_duration)
        data = pd.DataFrame({'amplitude': wave})
        data['time(ms)'] =  (np.arange(0,opts.sr*opts.audio_duration)*1/opts.sr)/(1e-3)
        data['class'] = [class_name]*data.shape[0]
        li.append(data)

    data = pd.concat(li)
    for ind, (data, file_path, class_name) in enumerate(zip(li, file_paths, class_names)): 
        fig = px.line(data, x='time(ms)', y='amplitude', color='class', 
                          template='ggplot2', title = "{:s}({:s})".format(file_path, class_name),
                         color_discrete_sequence=[px.colors.qualitative.D3[ind]], height = 400)
        
        fig['layout']['font']['size'] = 15
        pio.write_image(fig, output_dir+ class_name + '.svg', width = 980, height = 500)
        iplot(fig)








