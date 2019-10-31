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
import pandas as pd
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
init_notebook_mode(connected=True)
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

def get_meta_info(filename):
	infos = filename.split('-')
	return [int(info) for info in infos]

def input_to_target(opts):
	# audio files and their corresponding labels
	train_paths = [opts.data_path + "fold1/*.wav", opts.data_path + "fold2/*.wav"]
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
		_, class_id, _, _ = get_meta_info(train_file.split('/')[-1].strip('.wav'))
		# print("train_file={:s}, class_id={:d}".format(train_file, class_id))
		train_labels.append(int(class_id))
		class_names.append(CLASS_ID[int(class_id)])

	# csv storing information for training dataset
	train_files = pd.DataFrame({'train_file_paths': train_files, 
								'train_labels': train_labels,
								'class_names': class_names})
	print(train_files.head())
	return train_files

	

  


	# train_files = pd.DataFrame({'train_file_paths': train_files})
	# train_files['ID'] = train_files['train_file_paths'].apply(lambda x:x.split('/')[-1].split('.')[0])
	# train_files['ID'] = train_files['ID'].astype(int)
	# train_files = train_files.sort_values(by='ID')
	# test_files = glob.glob(test_path)









