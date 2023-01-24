#!/usr/bin/env python3
''' The main function '''
import argparse
import sys

import matplotlib.pyplot as plt
from cv2 import IMREAD_GRAYSCALE, imread

from src.image_recognition import get_grid
from src.solver import Board
from src.tools import draw_matrix, solution_img

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('task', help='Solve or generate')
parser.add_argument('-i', help='Image to solve')
args = parser.parse_args()

if args.task == 'generate':
    B = Board(9)
    B.generate_puzzle(1, to_remove=30)
    draw_matrix(B.grid, name='puzzle.png')
    # solution_img(B.solution(), B.grid, 'solution.png')
    sys.exit()
elif args.task == 'solve':
    img = imread(args.i, IMREAD_GRAYSCALE)
    grid = get_grid(img)
    print(grid)
    B = Board(9, grid=grid)
    solution_img(B.solution(), B.grid, 'solution.png')
    # sys.exit()