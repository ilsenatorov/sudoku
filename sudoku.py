#!/usr/bin/python3
from solver import Board
from image_recognition import get_grid
from keras.models import model_from_json
from cv2 import imread, IMREAD_GRAYSCALE
from tools import solution_img, draw_matrix
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from train_network import generate_dataset




task = sys.argv[1]

if task == 'generate':
    B = Board(9)
    B.generate_puzzle(1, to_remove=30)
    draw_matrix(B.grid, name='puzzle.png')
    solution_img(B.solution(), B.grid, 'solution.png')
    sys.exit()
elif task == 'solve':
    img_loc = sys.argv[2]
    img = imread(img_loc, IMREAD_GRAYSCALE)
    scaler = StandardScaler()
    classifier = KNeighborsClassifier(n_neighbors=3)
    x_train, y_train = generate_dataset(250)
    x_train = x_train.reshape(250,784)
    scaler.fit(x_train)
    classifier.fit(x_train, y_train)
    grid = get_grid(img, classifier, method='knn')
    B = Board(9, grid)
    print(B.grid)
    solution_img(B.solution(), B.grid, 'solution.png')