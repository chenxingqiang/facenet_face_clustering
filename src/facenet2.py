from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
from subprocess import Popen, PIPE

import numpy as np
import tensorflow as tf
from scipy import misc
from tensorflow.python.platform import gfile


def calculate_dist(embeddings1, embeddings2):
	assert (embeddings1.shape[0] == embeddings2.shape[0])
	assert (embeddings1.shape[1] == embeddings2.shape[1])
	diff = np.subtract( embeddings1, embeddings2 )
	dist = np.sum( np.square( diff ), 1 )
	return dist


def calculate_same(threshold, dist):
	predict_issame = np.less( dist, threshold )

	return predict_issame


def load_model(model):
	# Check if the model is a model directory (containing a metagraph and a checkpoint file)
	#  or if it is a protobuf file with a frozen graph
	model_exp = os.path.expanduser( model )
	if os.path.isfile( model_exp ):
		print( 'Model filename: %s' % model_exp )
		with gfile.FastGFile( model_exp, 'rb' ) as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString( f.read() )
			tf.import_graph_def( graph_def, name='' )
	else:
		print( 'Model directory: %s' % model_exp )
		meta_file, ckpt_file = get_model_filenames( model_exp )

		print( 'Metagraph file: %s' % meta_file )
		print( 'Checkpoint file: %s' % ckpt_file )

		saver = tf.train.import_meta_graph( os.path.join( model_exp, meta_file ) )
		saver.restore( tf.get_default_session(), os.path.join( model_exp, ckpt_file ) )


def get_model_filenames(model_dir):
	files = os.listdir( model_dir )
	meta_files = [s for s in files if s.endswith( '.meta' )]
	if len( meta_files ) == 0:
		raise ValueError( 'No meta file found in the model directory (%s)' % model_dir )
	elif len( meta_files ) > 1:
		raise ValueError( 'There should not be more than one meta file in the model directory (%s)' % model_dir )
	meta_file = meta_files[0]
	meta_files = [s for s in files if '.ckpt' in s]
	max_step = -1
	for f in files:
		step_str = re.match( r'(^model-[\w\- ]+.ckpt-(\d+))', f )
		if step_str is not None and len( step_str.groups() ) >= 2:
			step = int( step_str.groups()[1] )
			if step > max_step:
				max_step = step
				ckpt_file = step_str.groups()[0]
	return meta_file, ckpt_file


def prewhiten(x):
	mean = np.mean( x )
	std = np.std( x )
	std_adj = np.maximum( std, 1.0 / np.sqrt( x.size ) )
	y = np.multiply( np.subtract( x, mean ), 1 / std_adj )
	return y


def load_data(image_paths, do_random_flip, do_random_crop, image_size, do_prewhiten=True):
	nrof_samples = len( image_paths )
	images = np.zeros( (nrof_samples, image_size, image_size, 3) )
	for i in range( nrof_samples ):
		img = misc.imread( image_paths[i] )
		if img.ndim == 2:
			img = to_rgb( img )
		if do_prewhiten:
			img = prewhiten( img )
		img = crop( img, do_random_crop, image_size )
		img = flip( img, do_random_flip )
		images[i, :, :, :] = img
	return images


def crop(image, random_crop, image_size):
	if image.shape[1] > image_size:
		sz1 = int( image.shape[1] // 2 )
		sz2 = int( image_size // 2 )
		if random_crop:
			diff = sz1 - sz2
			(h, v) = (np.random.randint( -diff, diff + 1 ), np.random.randint( -diff, diff + 1 ))
		else:
			(h, v) = (0, 0)
		image = image[(sz1 - sz2 + v):(sz1 + sz2 + v), (sz1 - sz2 + h):(sz1 + sz2 + h), :]
	return image


def flip(image, random_flip):
	if random_flip and np.random.choice( [True, False] ):
		image = np.fliplr( image )
	return image


def to_rgb(img):
	w, h = img.shape
	ret = np.empty( (w, h, 3), dtype=np.uint8 )
	ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
	return ret


class ImageClass():
	"Stores the paths to images for a given class"

	def __init__(self, name, image_paths):
		self.name = name
		self.image_paths = image_paths

	def __str__(self):
		return self.name + ', ' + str( len( self.image_paths ) ) + ' images'

	def __len__(self):
		return len( self.image_paths )


def get_dataset(paths):
	dataset = []
	for path in paths.split( ':' ):
		path_exp = os.path.expanduser( path )
		classes = os.listdir( path_exp )
		classes.sort()
		nrof_classes = len( classes )
		for i in range( nrof_classes ):
			class_name = classes[i]
			facedir = os.path.join( path_exp, class_name )
			if os.path.isdir( facedir ):
				images = os.listdir( facedir )
				image_paths = [os.path.join( facedir, img ) for img in images]
				dataset.append( ImageClass( class_name, image_paths ) )

	return dataset


def store_revision_info(src_path, output_dir, arg_string):
	# Get git hash
	gitproc = Popen( ['git', 'rev-parse', 'HEAD'], stdout=PIPE, cwd=src_path )
	(stdout, _) = gitproc.communicate()
	git_hash = stdout.strip()

	# Get local changes
	gitproc = Popen( ['git', 'diff', 'HEAD'], stdout=PIPE, cwd=src_path )
	(stdout, _) = gitproc.communicate()
	git_diff = stdout.strip()

	# Store a text file in the log directory
	rev_info_filename = os.path.join( output_dir, 'revision_info.txt' )
	with open( rev_info_filename, "w" ) as text_file:
		text_file.write( 'arguments: %s\n--------------------\n' % arg_string )
		text_file.write( 'git hash: %s\n--------------------\n' % git_hash )
		text_file.write( '%s' % git_diff )
