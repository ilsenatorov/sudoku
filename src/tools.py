#!/usr/bin/python3
from os import listdir
from random import choice, randint

import numpy as np
from cv2 import IMREAD_GRAYSCALE, imread
from PIL import Image, ImageDraw, ImageFont

from .image_recognition import get_grid
from .solver import Board


def draw_empty_grid(size):
    ''' Draw an empty grid '''
    image = Image.new(mode='RGB', size=(size+3, size+3), color=(255,255,255))
    draw = ImageDraw.Draw(image)
    y_start = 0
    y_end = size+2
    x_start = 0
    x_end = size+2
    step_size = int(size/9)
    for x in range(1, size+2, step_size):
        line = ((x, y_start), (x, y_end))
        draw.line(line, fill=0)

    for y in range(1, size+2, step_size):
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill=0)

    for x in range(1, size+2, step_size*3): # Thick lines
        line = ((x-1, y_start), (x-1, y_end))
        draw.line(line, fill=0)
        line = ((x+1, y_start), (x+1, y_end))
        draw.line(line, fill=0)

    for y in range(1, size+2, step_size*3): # Thick lines
        line = ((x_start, y-1), (x_end, y-1))
        draw.line(line, fill=0)
        line = ((x_start, y+1), (x_end, y+1))
        draw.line(line, fill=0)
    return image


def draw_matrix(grid, name=False, size=450, frame=False):
    ''' Create the image from grid '''
    image = draw_empty_grid(size)
    step_size = int(image.width/9)
    draw = ImageDraw.Draw(image)
    font_path = "/usr/share/fonts/TTF/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 32)
    for i in range(9):
        for j in range(9):
            item = grid[i,j]
            w, h = draw.textsize(str(item))
            x = i*step_size + (50-w)/2
            y = j*step_size + (50-h)/2
            if item == 0:
                item = ' '
            draw.text((y,x), str(item), fill=(0,0,0), font=font)
    if frame:
        newimg = Image.new(mode='L', size=(size+13, size+13), color=255)
        newimg.paste(image, (5,5))
        image = newimg
    if name:
        image.save(name)
    return image


def solution_img(solved, puzzle, name):
    '''Creates an image with puzzle in black and solutions in red '''
    image = draw_matrix(puzzle)
    step_size = int(image.width/9)
    draw = ImageDraw.Draw(image)
    font_path = "/usr/share/fonts/TTF/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 32)

    for i in range(9):
        for j in range(9):
            if solved[i,j] != puzzle[i,j]:
                item = solved[i,j]
                x = i*step_size + 12
                y = j*step_size + 7
                if item == 0:
                    item = ' '
                draw.text((y,x),str(item), fill=(255,0,0), font=font)
    if name:
        image.save(name)



def recover_dataset(picfolder):
    ''' From pictures of sudoku grids create the dataset '''
    res = []
    pics = sorted(listdir(picfolder))
    for pic in pics:
        img = imread(picfolder+pic, IMREAD_GRAYSCALE)
        digits = get_grid(img, 'a', 'b', no_predict=True)
        res.append(digits)
    return np.asarray(res).reshape(81*len(pics),28,28)
