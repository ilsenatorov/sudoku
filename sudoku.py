#!/usr/bin/python3
from solver import Board
from image_recognition import get_grid
from keras.models import model_from_json
from cv2 import imread, IMREAD_GRAYSCALE
from tools import solution_img

# TODO get a better dataset
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")


img_loc = 'board.jpeg'
img = imread(img_loc, IMREAD_GRAYSCALE)

grid = get_grid(img, model)

B = Board(9, grid)
print(B.grid)
# B.prepare(1)
solution_img(B.solution(), B.grid, 'test.png')