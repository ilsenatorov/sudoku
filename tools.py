#!/usr/bin/python3
from PIL import Image, ImageFont, ImageDraw
font = ImageFont.truetype('APHont-Bold_q15c.ttf', size=45)


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