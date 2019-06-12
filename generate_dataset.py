#!/usr/bin/python3
''' Generates n pictures with a full sudoku grid and their actual values for training '''
from tools import save_generated_dataset
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('picfolder', help='Folder in which to put the images')
parser.add_argument('number', help='How many pictures to generate')
args = parser.parse_args()

save_generated_dataset(args.picfolder, args.number)