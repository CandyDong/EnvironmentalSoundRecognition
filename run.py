#!/usr/bin/env python

from options import get_options
import pprint as pp
from preprocess import input_to_target, plot_mel_spectrogram, plot_time_amplitude

def run(opts):
	# Pretty print the run args
	pp.pprint(vars(opts))
	
	train_file_df = input_to_target(opts)
	# plot_mel_spectrogram(train_file_df, opts)
	plot_time_amplitude(train_file_df, opts)
	


if __name__ == "__main__":
	run(get_options())