from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import facenet2
import numpy as np


def evaluate(embeddings):
	# Calculate evaluation metrics
	thresholds = 0.8  # we have proof that the best threshold of our dataset is about 0.76 to 0.8
	embeddings1 = embeddings[0::2]
	embeddings2 = embeddings[1::2]
	dist = facenet2.calculate_dist( embeddings1, embeddings2 )
	predict_issame = facenet2.calculate_same( thresholds, dist )
	return predict_issame


def get_paths(allface_dir, pairs, file_ext):
	nrof_skipped_pairs = 0
	path_list = []
	pair_list = []
	for pair in pairs:
		if len( pair ) == 4:
			path0 = os.path.join( allface_dir, 'allface', pair[0] + '_' + '%04d' % int( pair[1] ) + '.' + file_ext )
			path1 = os.path.join( allface_dir, 'allface', pair[2] + '_' + '%04d' % int( pair[3] ) + '.' + file_ext )
		if os.path.exists( path0 ) and os.path.exists( path1 ):  # Only add the pair if both paths exist
			path_list += (path0, path1)
			pair_list.append( [pair[0] + '_' + '%04d' % int( pair[1] ), pair[2] + '_' + '%04d' % int( pair[3] )] )
		else:
			nrof_skipped_pairs += 1
	if nrof_skipped_pairs > 0:
		print( 'Skipped %d image pairs' % nrof_skipped_pairs )

	return path_list, pair_list


def read_pairs(pairs_filename):
	pairs = []
	with open( pairs_filename, 'r' ) as f:
		for line in f.readlines()[1:]:
			pair = line.strip().split()
			pairs.append( pair )
	return np.array( pairs )
