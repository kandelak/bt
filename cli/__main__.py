import argparse
import os.path

# TODO add advanced path handling (os independent) using os.path or pathlib (search for "Working wiht paths in python
#  best practices" )

parser = argparse.ArgumentParser(description='Process input file with resolution and method')
parser.add_argument('-i', '--input_file', required=False, help='Folder location of aerial imagery')
parser.add_argument('-r', '--resolution', required=False, help='Resolution in cm')
parser.add_argument('-m', '--method', required=False, help='Method of resample')

args = parser.parse_args()

input_file, method, resolution = args.input_file, args.method, args.resolution
