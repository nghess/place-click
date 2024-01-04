import cv2
import numpy as np


# Get mouse location with cv2 mouse events
m_loc = [0,0]

def mouse_loc(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        m_loc[0] = x
        m_loc[1] = y

# 2d Gaussian helper function
def gaussian_2d(x, y, mu_x=0, mu_y=0, sigma_x=5, sigma_y=5):

    return 1 / (2 * np.pi * sigma_x * sigma_y) * np.exp(-((x - mu_x)**2 / (2 * sigma_x**2) + (y - mu_y)**2 / (2 * sigma_y**2)))

# Gaussian Maze class
class gridMaze():
    def __init__(self, maze_bounds, maze_dims, std=10, sparsity=0):
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

        # Generate visited targets list
        self.visited = [[] for item in range(len(self.targets))]

    def check_location(self, m_loc):
        for ii in range(len(self.targets)):
            if self.cells[ii][0][0] < m_loc[0] < self.cells[ii][1][0]:
                if self.cells[ii][0][1] < m_loc[1] < self.cells[ii][1][1]:
                    if self.targets[ii] == 1:
                        self.targets[ii] = 0
                        cell_center_x = (self.cells[ii][0][0] + self.cells[ii][1][0]) // 2
                        cell_center_y = (self.cells[ii][0][1] + self.cells[ii][1][1]) // 2
                        self.visited[ii] = cell_center_x, cell_center_y


            
"""
Test
"""

# Generate maze and canvas
maze = gridMaze([1200,800], [12,8], std=50, sparsity=.3)
canvas = np.zeros([800,1200,3], dtype=np.uint8)
draw = True

# Start mouse callbacks
cv2.namedWindow('Gaussian Maze')
cv2.setMouseCallback('Gaussian Maze', mouse_loc)

# Draw
while draw:
    for ii in range(len(maze.cells)):

        # Color cells by target status
        target = maze.targets[ii]
        cv2.rectangle(canvas, maze.cells[ii][0], maze.cells[ii][1], (int(255*target),0,0), thickness=-1)  
        
        # Check if mouse is inside a target rectangle
        maze.check_location(m_loc)

    # Mark visited targets
    for ii in range(len(maze.visited)):
                if np.sum(maze.visited[ii]) > 0:
                    cv2.circle(canvas, maze.visited[ii], int(50*maze.density[ii]), (0,0,255), thickness=-1)
    
    # Current mouse location
    cv2.putText(canvas, f"{m_loc[0]},{m_loc[1]}", (m_loc[0]+20,m_loc[1]+20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.3, color=(255,255,255))

    # If targets have all been visited, generate new maze.
    if np.mean(maze.targets) == 0:
        maze = gridMaze([1200,800], [12,8], std=50, sparsity=.3)
            
    cv2.imshow("Gaussian Maze", canvas)
    if cv2.waitKey(1) == ord('q'):
        break


