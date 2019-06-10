#!/usr/bin/python3
from PIL import Image, ImageFont, ImageDraw
from cv2 import imread, IMREAD_GRAYSCALE
from random import randint
import numpy as np
from solver import Board
from image_recognition import get_grid
from os import listdir
font = ImageFont.truetype('fonts/APHont-Regular_q15c.ttf', size=45)


def draw_empty_grid(size):
    ''' Draw an empty grid '''
    image = Image.new(mode='RGB', size=(size, size), color=(255,255,255))
    draw = ImageDraw.Draw(image)
    y_start = 0
    y_end = image.height
    x_start = 0
    x_end = image.width
    step_size = int(image.width/9)
    for x in range(0, image.width, step_size):
        line = ((x, y_start), (x, y_end))
        draw.line(line, fill=0)

    for y in range(0, image.height, step_size):
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill=0)

    for x in range(0, image.width, step_size*3): # Thick lines
        line = ((x-1, y_start), (x-1, y_end))
        draw.line(line, fill=0)
        line = ((x+1, y_start), (x+1, y_end))
        draw.line(line, fill=0)

    for y in range(0, image.height, step_size*3): # Thick lines
        line = ((x_start, y-1), (x_end, y-1))
        draw.line(line, fill=0)
        line = ((x_start, y+1), (x_end, y+1))
        draw.line(line, fill=0)
    return image


def draw_matrix(grid, name=False, size=450):
    ''' Create the image from grid '''
    image = draw_empty_grid(size)
    step_size = int(image.width/9)
    draw = ImageDraw.Draw(image)
    for i in range(9):
        for j in range(9):
            item = grid[i,j]
            x = i*step_size + 10
            y = j*step_size + 5
            if item == 0:
                item = ' '
            draw.text((x,y),str(item), font=font, fill=(0,0,0))
    if name:
        image.save(name)
    return image


def solution_img(solved, puzzle, name):
    '''Creates an image with puzzle in black and solutions in red '''
    image = draw_matrix(puzzle)
    step_size = int(image.width/9)
    draw = ImageDraw.Draw(image)
    for i in range(9):
        for j in range(9):
            if solved[i,j] != puzzle[i,j]:
                item = solved[i,j]
                x = i*step_size + 12
                y = j*step_size + 7
                if item == 0:
                    item = ' '
                draw.text((x,y),str(item), font=font, fill=(255,0,0))
    if name:
        image.save(name)


def generate_image():
    ''' Generates a 28x28 image with a random digit 1-9, with slight degree of randomness'''
    background = 0
    image = Image.new(mode='L', size=(28, 28), color=background)
    txt = Image.new(mode='L', size=(28,28), color=background)
    draw = ImageDraw.Draw(txt)
    x = randint(4,6)
    y = randint(2,4)
    num = randint(1,9)
    fontsize = randint(26,28)
    fill = randint(220,255)
    angle = randint(-5,5)
    # font = choice(listdir("fonts/"))
    font = ImageFont.truetype('fonts/APHont-Bold_q15c.ttf', size=fontsize)
    draw.text((x,y), str(num), font=font, fill=fill)
    txt = txt.rotate(angle, expand=0)
    image.paste(txt)
    return np.asarray(image.getdata()), int(num)

def generate_dataset(number):
    ''' Generate a dataset of size _number_ with random images of digits '''
    x_train = np.zeros((number, 784), int)
    y_train = np.zeros(number, int)
    for i in range(number):
        x, y = generate_image()
        x_train[i] = x
        y_train[i] = y
    return x_train.reshape(number, 28, 28), y_train

def save_generated_dataset(folder, number):
    y = np.zeros((number, 81), int)
    for i in range(number):
        picname = folder + 'orig/grid' + str(i).zfill(4) + '.png'
        B = Board(9)
        grid = B.solution()
        draw_matrix(grid, picname)
        y[i] = grid.reshape(81)
    np.savetxt(folder+'data.csv', y, delimiter=',', fmt='%d')



def recover_dataset(picfolder):
    ''' From pictures of sudoku grids create the dataset '''
    res = []
    pics = sorted(listdir(picfolder))
    for pic in pics:
        img = imread(picfolder+pic, IMREAD_GRAYSCALE)
        digits = get_grid(img, 'a', 'b', no_predict=True)
        res.append(digits)
    return np.asarray(res).reshape(8100,28,28)
