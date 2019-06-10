#!/usr/bin/python3
from solver import Board
from image_recognition import get_grid
from keras.models import model_from_json
from cv2 import imread, IMREAD_GRAYSCALE
from tools import solution_img, draw_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from train_network import generate_dataset

import sys
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('task', help='Solve or generate')
parser.add_argument('-i', help='Image to solve')
parser.add_argument('--method', help='Which ML algorithm to use - cnn or knn')
parser.add_argument('--model', help='Location of the model.json and model.h5 files for CNN')
args = parser.parse_args()

if args.task == 'generate':
    B = Board(9)
    B.generate_puzzle(1, to_remove=30)
    draw_matrix(B.grid, name='puzzle.png')
    # solution_img(B.solution(), B.grid, 'solution.png')
    sys.exit()
elif args.task == 'solve':
    img = imread(args.i, IMREAD_GRAYSCALE)
    if args.method == 'knn':
        scaler = StandardScaler()
        model = KNeighborsClassifier(n_neighbors=3)
        x_train, y_train = generate_dataset(250)
        x_train = x_train.reshape(250,784)
        scaler.fit(x_train)
        model.fit(x_train, y_train)
    elif args.method == 'cnn':
        json_file = open(args.model + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(args.model +'.h5')
    grid = get_grid(img, model, method=args.method)
    B = Board(9, grid)
    print(B.grid)
    # solution_img(B.solution(), B.grid, 'solution.png')
    # sys.exit()