from __future__ import print_function

import argparse
import os
import shutil
import sys

import networkx as nx
from networkx.algorithms.approximation import clique_removal


# file_path1 = "./faces_random/allface/"
# file_path2 = "./faces_random/"


def read_predict(predict_dir):
	all_set = []
	f1 = open( predict_dir + "predict.txt", 'r' )
	for line in f1:
		# print(line.strip('\n'))
		all_set.append(
			(line.strip( '\n' ).split( '\t' )[0].strip( ' ' ), line.strip( '\n' ).split( '\t' )[1].strip( ' ' ), 1) )
	f1.close()
	# f2 = open( "edgelist_file.txt", 'w+' )
	# for set in all_set:
	#	print( (set) )
	#	print( set[0], ' ', set[1], ' ', str( set[2] ), file=f2 )
	# f2.close()
	return all_set


# def find_community(graph, k):
#	return list( nx.k_clique_communities( graph, k ) )

def mk_classify(file_path1, file_path2, predict_dir):
	G = nx.Graph()
	all_set = read_predict( predict_dir )
	for set in all_set:
		G.add_edge( set[0], set[1] )
	# print( G.nodes() )
	# print( G.edges() )
	# print( G.number_of_edges() )
	# iset=max_clique( G )
	maxiset, cliqueset = clique_removal( G )
	# print(maxiset)
	# print( len( cliqueset ) )
	for i in range( 0, len( cliqueset ) ):
		os.mkdir( file_path2 + 'face' + str( i ) )
		targetpath = os.path.join( file_path2 + 'face' + str( i ) )
		print( targetpath )
		for graph in cliqueset[i]:
			sourcepath = os.path.join( file_path1 + graph + '.png' )
			move_fileto( sourcepath, targetpath )


def move_fileto(sourceDir, targetDir):
	shutil.copy( sourceDir, targetDir )


def parse_arguments(argv):
	parser = argparse.ArgumentParser()

	parser.add_argument( '--allface_dir', type=str,
	                     help='Path to the data directory containing aligned Allface face patches.' )
	parser.add_argument( '--allface_class_dir', type=str,
	                     help='path for save the classified graphs' )
	parser.add_argument( '--predict_dir', type=str,
	                     help='path to predict result file' )
	return parser.parse_args( argv )


def main():
	path1 = parse_arguments( sys.argv[1:] ).allface_dir
	path2 = parse_arguments( sys.argv[1:] ).allface_class_dir
	path3 = parse_arguments( sys.argv[1:] ).predict_dir
	mk_classify( path1, path2, path3 )


if __name__ == '__main__':
	main()
