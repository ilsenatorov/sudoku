#!/usr/bin/python3
''' Class for generating sudoku puzzles and solving them.'''
import numpy as np
from random import randint, shuffle
from copy import deepcopy

class Board(object):
    ''' The main Board class - initialised with board of 0 '''
    def __init__(self, size, grid=False):
        if isinstance(grid, bool):
            self.grid = np.zeros((size,size), int)
        else:
            self.grid = grid
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