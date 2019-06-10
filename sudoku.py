#!/usr/bin/python3
from solver import Board
from image_recognition import get_grid
from keras.models import model_from_json
from cv2 import imread, IMREAD_GRAYSCALE
from tools import solution_img, draw_matrix
import sys

task = sys.argv[1]

if task == 'generate':
    B = Board(9)
    B.generate_puzzle(1, to_remove=30)
    draw_matrix(B.grid, name='puzzle.png')
    solution_img(B.solution(), B.grid, 'solution.png')
    sys.exit()
elif task == 'solve':
    img_loc = sys.argv[2]
    model_name = sys.argv[3]
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_name + '.h5')
    img = imread(img_loc, IMREAD_GRAYSCALE)
    grid = get_grid(img, model)
    B = Board(9, grid)
    print(B.grid)
    solution_img(B.solution(), B.grid, 'solution.png')