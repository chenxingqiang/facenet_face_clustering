from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import math
import os
import sys

import allface
import facenet2
import numpy as np
import tensorflow as tf


def main(args):
	with tf.Graph().as_default():
		with tf.Session() as sess:
			# Read the file containing the pairs used for testing
			pairs = allface.read_pairs( os.path.expanduser( args.allface_pairs ) )
			# print(pairs)
			# Get the paths for the corresponding images
			paths, pair2 = allface.get_paths( os.path.expanduser( args.allface_dir ), pairs, args.allface_file_ext )
			print( paths )
			# Load the model
			facenet2.load_model( args.model )

			# Get input and output tensors
			images_placeholder = tf.get_default_graph().get_tensor_by_name( "input:0" )
			embeddings = tf.get_default_graph().get_tensor_by_name( "embeddings:0" )
			phase_train_placeholder = tf.get_default_graph().get_tensor_by_name( "phase_train:0" )

			# image_size = images_placeholder.get_shape()[1]  # For some reason this doesn't work for frozen graphs
			image_size = args.image_size
			embedding_size = embeddings.get_shape()[1]

			# Run forward pass to calculate embeddings
			print( 'Running forward pass on allface images' )
			batch_size = args.allface_batch_size
			nrof_images = len( paths )
			nrof_batches = int( math.ceil( 1.0 * nrof_images / batch_size ) )
			emb_array = np.zeros( (nrof_images, embedding_size) )
			for i in range( nrof_batches ):
				start_index = i * batch_size
				end_index = min( (i + 1) * batch_size, nrof_images )
				paths_batch = paths[start_index:end_index]
				images = facenet2.load_data( paths_batch, False, False, image_size )
				feed_dict = {images_placeholder: images, phase_train_placeholder: False}
				emb_array[start_index:end_index, :] = sess.run( embeddings, feed_dict=feed_dict )

			predict_value = allface.evaluate( emb_array )
			f = open( args.allface_dir + "predict.txt", 'w+' )
			print( len( pair2 ), len( predict_value ) )
			for i in range( 0, len( pair2 ) ):
				if predict_value[i] == True:
					print( pair2[i][0], '	', pair2[i][1], '	', predict_value[i], file=f )
			f.close()


def parse_arguments(argv):
	parser = argparse.ArgumentParser()

	parser.add_argument( 'allface_dir', type=str,
	                     help='Path to the data directory containing aligned Allface face patches.' )
	parser.add_argument( '--allface_batch_size', type=int,
	                     help='Number of images to process in a batch in the Allface classify set.', default=100 )
	parser.add_argument( 'model', type=str,
	                     help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf '
	                          '(.pb) file' )
	parser.add_argument( '--image_size', type=int,
	                     help='Image size (height, width) in pixels.', default=160 )
	parser.add_argument( '--allface_pairs', type=str,
	                     help='The file containing the pairs to use for validation.', default='data/pairs.txt' )
	parser.add_argument( '--allface_file_ext', type=str,
	                     help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'] )
	# parser.add_argument('--allface_nrof_folds', type=int,
	#                    help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
	return parser.parse_args( argv )


if __name__ == '__main__':
	main( parse_arguments( sys.argv[1:] ) )
