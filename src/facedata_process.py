from __future__ import print_function

import argparse
import os
import os.path
import random
import sys
from itertools import combinations


def rename(path):
	file_array = []
	nrof_photo = 0;
	# path="./faces_random/";
	filelist = os.listdir( path );
	# print filelist
	for files in filelist:
		Olddir = os.path.join( path, files );
		# print Olddir
		if os.path.isdir( Olddir ):
			filename = os.path.splitext( files )[0];
			# print filename
			filelist2 = os.listdir( Olddir )
			# print filelist2
			count = 1;
			for file in filelist2:
				# print file
				file_array.append( file )
				Olddir2 = os.path.join( path, files, file )
				Newdir = os.path.join( path, files, 'face_' + '%04d' % count + '.png' )
				# print  Olddir2
				# print  Newdir
				count += 1;
				nrof_photo += 1;
				os.rename( Olddir2, Newdir )
	# print nrof_photo
	# print file_array
	return file_array


# rename()

def make_pairs(path):
	pair_members = rename( path );
	notsame_pairs = list( combinations( pair_members, 2 ) );
	print( len( notsame_pairs ) )
	f = open( path + 'pairs_face_rand.txt', 'w+' )
	print( '1', str( len( notsame_pairs ) ), file=f )
	for k in range( 0, 10 ):
		slect_notsame_pairs = sorted( random.sample( notsame_pairs, int( round( len( notsame_pairs ) / 10 ) ) ) )
		for notsame_pair in slect_notsame_pairs:
			print( notsame_pair[0].split( '_' )[0], '	', notsame_pair[0].split( '_' )[1].split( '.' )[0], '	', notsame_pair[1].split( '_' )[0], '	', notsame_pair[1].split( '_' )[1].split( '.' )[0], file=f )
	f.close()


def parse_arguments(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument( 'faces_dir', type=str,
	                     help='Path to the data directory containing aligned face patches.' )
	return parser.parse_args( argv )


def main():
	make_pairs( parse_arguments( sys.argv[1:] ).faces_dir )
	print( parse_arguments( sys.argv[1:] ).faces_dir )


if __name__ == '__main__':
	main()
