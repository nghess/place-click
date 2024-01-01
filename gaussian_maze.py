import cv2
import numpy as np

"""
draw equally spaced grid within bounds
choose number of active cells
set seed
"""

def gaussian_2d(x, y, mu_x=0, mu_y=0, sigma_x=5, sigma_y=5):

    return 1 / (2 * np.pi * sigma_x * sigma_y) * np.exp(-((x - mu_x)**2 / (2 * sigma_x**2) + (y - mu_y)**2 / (2 * sigma_y**2)))


class gridMaze():
    def __init__(self, maze_bounds, maze_dims, std=10, sparsity=0, contiguous=True):
        assert isinstance(maze_bounds, (tuple, list)), "Arena dims argument must be tuple or list."
        assert isinstance(maze_dims, (tuple, list)), "Maze dims argument must be tuple or list."

        self.bounds = maze_bounds
        self.shape = maze_dims
        self.density = []
        self.labels = []
        
        cellsize_x = maze_bounds[0] // maze_dims[0]
        cellsize_y = maze_bounds[1] // maze_dims[1]

        # Define corners of each cell in pixel space
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

        # Set this to mouse click or random
        focus_x = np.random.randint(0,self.shape[1])
        focus_y = np.random.randint(0,self.shape[0])

        # Generate Gaussian
        for y in range(self.shape[1]):
            for x in range(self.shape[0]):
                cell = gaussian_2d(x, y, mu_x=focus_x, mu_y=focus_y, sigma_x=std, sigma_y=std)
                self.labels.append(f"{x},{y}")
                self.density.append(cell)

        # Normalize    
        self.density -= np.min(self.density)
        self.density /= np.max(self.density)
        self.density -= sparsity

        # Generate target list based on probability density

        self.targets = np.array(np.random.rand(len(self.density)))
        self.targets = (self.targets <= self.density)
            

"""
Test
"""

test = gridMaze([1200,800], [24,16], std=50, sparsity=.3)
canvas = np.zeros([800,1200,3], dtype=np.uint8)

# Draw
for ii in range(len(test.cells)):

    #Color cells by probability
    #density = test.density[ii]
    #cv2.rectangle(canvas, test.cells[ii][0], test.cells[ii][1], (255*density,0,0), thickness=-1)

    #Color cells by target status
    target = test.targets[ii]
    cv2.rectangle(canvas, test.cells[ii][0], test.cells[ii][1], (int(255*target),0,0), thickness=-1)

    #Label x,y for sanity check purposes
    #label = test.labels[ii]
    #label_coord = (np.asarray(test.cells[ii][0]) + np.asarray(test.cells[ii][1])) // 2
    #cv2.putText(canvas, label, label_coord, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.3, color=(0,0,255))

cv2.imshow("grid", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

