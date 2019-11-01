#!/usr/bin/env python
import numpy as np
from options import get_options
import pprint as pp
from preprocess import input_to_target, plot_mel_spectrogram, plot_time_amplitude, load_audio_feature
from models.SVM import SVMModel

def _gen_dataset(df, opts):
	X, y = None, []

	for file_path, label in zip(df['file_paths'], df['labels']):
		mfccs, chroma, mel, contrast = load_audio_feature(file_path, opts)
		# print("S.shape:" + str(S.shape))
		if X is None:
			X = np.hstack([mfccs, chroma, mel, contrast])
		else:
			X = np.vstack([X, np.hstack([mfccs, chroma, mel, contrast])])
		y += [label]
	
	# assert(X.shape[1] == len(y))
	return np.array(X), np.array(y)

def run(opts):
	# Pretty print the run args
	pp.pprint(vars(opts))
	
	train_file_df, test_file_df = input_to_target(opts)
	# plot_mel_spectrogram(train_file_df, opts)
	# plot_time_amplitude(train_file_df, opts)

	X_train, y_train = _gen_dataset(train_file_df, opts)
	# print("Train dataset size = {:s}".format(str(X_train.shape)))
	# print("Train label size = {:s}".format(str(y_train.shape)))
	X_test, y_test = _gen_dataset(test_file_df, opts)
	# print("Test dataset size = {:s}".format(str(X_test.shape)))
	svm = SVMModel(opts)
	svm.run(X_train, y_train)
	confusion_matrix, classification_report, acc = svm.evaluate(X_test, y_test)
	print("confusion_matrix:")
	print(str(confusion_matrix))
	print("classification_report:")
	print(str(classification_report))
	print("accuracy={:0.3f}".format(acc))
	


if __name__ == "__main__":
	run(get_options())