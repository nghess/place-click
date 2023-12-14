import cv2
import numpy as np

"""
draw equally spaced grid within bounds
choose number of active cells
set seed
"""

class gridMaze():
    def __init__(self, maze_bounds, maze_dims, contiguous=True):
        assert type(maze_bounds) == tuple or list, "Arena dims argument must be tuple or list"
        assert type(maze_dims) == tuple or list, "Maze dims argument must be tuple or list"
        #assert maze_dims[0] >= 2, "Maze must be at least 2 rows tall."
        #assert maze_dims[1] >= 3, "Maze must be at least 3 columns wide."
        self.bounds = maze_bounds
        self.shape = maze_dims
        self.cellsize = maze_bounds[0] // maze_dims[0]

        xcoord = 0
        ycoord = 0
        coords = []
        for ii in range(self.shape[0]):
            coords.append([xcoord, ycoord])
            xcoord += self.cellsize
            for jj in range(self.shape[1]):
                ycoord += self.cellsize


