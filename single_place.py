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

        self.bounds = maze_bounds
        self.shape = maze_dims
        
        cellsize_x = maze_bounds[0] // maze_dims[0]
        cellsize_y = maze_bounds[1] // maze_dims[1]


        # Generate Grid
        xcoord = 0
        ycoord = 0
        coords = []
        for x in range(self.shape[1]):
            for y in range(self.shape[0]):
                coords.append(([xcoord, ycoord],[xcoord+cellsize_x, ycoord+cellsize_y]))
                xcoord += cellsize_x
            ycoord += cellsize_y
            xcoord = 0

        self.cells = coords


"""
Test
"""
test = gridMaze([1200,800], [6,4])
canvas = np.zeros([800,1200,3], dtype=np.uint8)

# Draw

for ii in range(len(test.cells)):
    cv2.rectangle(canvas, test.cells[ii][0], test.cells[ii][1], (255,255,255), thickness=2)

# Pull out a single cell for neighbor testing
cv2.rectangle(canvas, test.cells[8][0], test.cells[8][1], (255,0,0), thickness=-1)  # Cell 8

cv2.imshow("grid", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
