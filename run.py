#!/usr/bin/env python

from options import get_options
import pprint as pp
from preprocess import input_to_target

def run(opts):
	# Pretty print the run args
	pp.pprint(vars(opts))
	
	input_to_target(opts)
	


if __name__ == "__main__":
	run(get_options())