#!/usr/bin/env python

from options import get_options
import pprint as pp
from preprocess import input_to_target, plot_mel_spectrogram, plot_time_amplitude, load_audio_spectrogram
from models.SVM import SVMModel

def _gen_dataset(df, opts):
	X, y = [], []
	for file_path, label in zip(df['file_paths'], df['labels']):
		X.append(load_audio_spectrogram(file_path, opts))
		y.append(label)
	return X, y

def run(opts):
	# Pretty print the run args
	pp.pprint(vars(opts))
	
	train_file_df, test_file_df = input_to_target(opts)
	# plot_mel_spectrogram(train_file_df, opts)
	# plot_time_amplitude(train_file_df, opts)
	X_train, y_train = _gen_dataset(train_file_df, opts)
	print("Train dataset size = {:d}".format(len(X_train)))
	X_test, y_test = _gen_dataset(test_file_df, opts)
	print("Test dataset size = {:d}".format(len(X_test)))
	svm = SVMModel(X_train, y_train, X_test, y_test, opts)
	svm.run()
	confusion_matrix, classification_report = svm.evaluate(X_test, y_test)
	print("confusion_matrix:")
	print(str(confusion_matrix))
	print("classification_report:")
	print(str(classification_report))
	


if __name__ == "__main__":
	run(get_options())