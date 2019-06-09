#!/usr/bin/python3
''' Class for generating sudoku puzzles and solving them.
TODO - turn board into image for easier visual comprehension
'''
import numpy as np
from random import randint, shuffle
from copy import deepcopy
from PIL import Image, ImageFont, ImageDraw
import profile

class Board(object):
    ''' The main Board class - initialised with board of 0 '''
    def __init__(self, size):
        self.grid = np.zeros((size,size), int)
        self.size = size

    def __getitem__(self, pos):
        i, j = pos
        return self.grid[i,j]

    def __setitem__(self, pos, val):
        i, j = pos
        self.grid[i,j] = val

    def __str__(self):
        res = ' '
        for i in range(self.size):
            for j in range(self.size-1):
                if self.grid[i,j] == 0:
                    res += ' |'
                else:
                    res += str(self.grid[i,j]) + '|'
            if self.grid[i, self.size-1] == 0:
                res += ' \n '
            else:
                res += str(self.grid[i, self.size-1]) + '\n'
        return res

    def _unique(self, subset):
        '''Check if subset is unique - no repeats'''
        return (np.unique(subset) == self.size)

    def isvalid(self):
        '''Check if grid is valid'''
        for i in range(self.size):
            if not self._unique(self.grid[i]) or not self._unique(self.grid[:i]):
                return False
        for i in range(0, self.size, 3):
            for j in range(0, self.size, 3):
                if not self._unique(self.grid[i:i+3,j:j+3]):
                    return False

    def isfull(self):
        '''Check if empty cells remaining'''
        if np.isin(0, self.grid):
            return False
        else:
            return True

    def empty_cell(self):
        '''Finds nearest empty cell'''
        flat = np.where(self.grid == 0)
        if flat[0].size == 0:
            return False
        i, j = flat
        return i[0], j[0]

    def _subgrid(self, i, j):
        '''Returns the top left cell of subgrid coordinates for i, j'''
        return (i - (i % 3)), (j - (j % 3))

    def solve(self):
        '''Implement backtracking search to fill up all the cells'''
        st = self.empty_cell()
        if st:
            i, j = st
        else:
            return
        r = list(range(1,10))
        shuffle(r)
        for val in r:
            if val not in self.grid[i] and val not in self.grid[:,j]:
                sub_i, sub_j = self._subgrid(i,j)
                if val not in self.grid[sub_i:sub_i+3, sub_j:sub_j+3]:
                    self.grid[i,j] = val
                    if self.isfull():
                        return True
                    else:
                        if self.solve():
                            return True
        self.grid[i,j] = 0

    def solution(self):
        '''Returns the solution without changing the grid'''
        cp = deepcopy(self)
        cp.solve()
        return cp.grid

    def prepare(self, attempts, checks=1, to_remove=False):
        ''' Prepares the puzzle '''
        self.solve()
        if not to_remove:
            to_remove = 10000
        while attempts > 0:
            if to_remove == 0:
                return
            i = randint(0,self.size - 1)
            j = randint(0,self.size - 1)
            while self.grid[i,j] == 0:
                i = randint(0,self.size - 1)
                j = randint(0,self.size - 1)
            val = self.grid[i,j]
            self.grid[i,j] = 0
            cp = deepcopy(self)
            if not np.array_equal(cp.solution(), self.solution()):
                to_remove += 1
                self.grid[i,j] = val
                attempts -= 1
            to_remove -= 1

    def solution_img(self, name):
        solved = self.solution()
        image = draw_matrix(self.grid)
        font = ImageFont.truetype('Hack.ttf', size=45)
        step_size = int(image.width/9)
        draw = ImageDraw.Draw(image)
        for i in range(9):
            for j in range(9):
                if solved[i,j] != self.grid[i,j]:
                    item = solved[i,j]
                    x = i*step_size + 10
                    y = j*step_size
                    if item == 0:
                        item = ' '
                    draw.text((x,y),str(item), font=font, fill=(255,0,0))
        if name:
            image.save(name)




def draw_matrix(grid, name=False, size=450):
    image = draw_empty_grid(size)
    font = ImageFont.truetype('Hack.ttf', size=45)
    step_size = int(image.width/9)
    draw = ImageDraw.Draw(image)
    for i in range(9):
        for j in range(9):
            item = grid[i,j]
            x = i*step_size + 10
            y = j*step_size
            if item == 0:
                item = ' '
            draw.text((x,y),str(item), font=font, fill=(0,0,0))
    if name:
        image.save(name)
    return image


def draw_empty_grid(size):
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




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--difficulty', help='Number of attempts, higher value corresponds to higher difficulty, defaults to 2', type=int)
    parser.add_argument('--to_remove', help='Number of empty cells to have', type=int)
    parser.add_argument('--solution', help='To print the correct solution, off by default', action='store_true')
    parser.add_argument('--size', help='Size of the board, has to be divisible by 3, defaults to 9', action='store_true')
    args = parser.parse_args()
    if args.difficulty:
        dif = args.difficulty
    else:
        dif = 2
    B = Board(9)
    B.prepare(2, to_remove=50)
    draw_matrix(B.grid, 'puzzle.png')
    if args.solution:
        B.solution_img('solution.png')