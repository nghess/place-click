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
    def __init__(self, px_bounds, grid_dims, std=10, sparsity=0):

        assert isinstance(px_bounds, (tuple, list)), "Arena dims argument must be tuple or list."
        assert isinstance(grid_dims, (tuple, list)), "Maze dims argument must be tuple or list."

        # Grid properties
        self.bounds = px_bounds
        self.shape = grid_dims
        self.labels = []
        self.cellsize_x = px_bounds[0] // grid_dims[0]
        self.cellsize_y = px_bounds[1] // grid_dims[1]

        # Gaussian properties 
        self.sparsity = sparsity
        self.probabilities = []
        self.std_dev = std

        # Select random cell as focus of Gaussian
        self.focus_x = np.random.randint(0,self.shape[1])
        self.focus_y = np.random.randint(0,self.shape[0])

        # Initialize cell coords
        xcoord = 0
        ycoord = 0
        coords = []

        # Generate upper and lower corner pixel coordinates for each cell
        for x in range(self.shape[1]):
            for y in range(self.shape[0]):
                coords.append(([xcoord, ycoord],[xcoord+self.cellsize_x, ycoord+self.cellsize_y]))
                xcoord += self.cellsize_x
            ycoord += self.cellsize_y
            xcoord = 0
        self.cells = coords

        # Generate Gaussian
        for y in range(self.shape[1]):
            for x in range(self.shape[0]):
                cell = gaussian_2d(x, y, mu_x=self.focus_x, mu_y=self.focus_y, sigma_x=std, sigma_y=std)
                self.labels.append(f"{x},{y}")
                self.probabilities.append(cell)

        # Normalize probabilities between 0 and 1, then adjust ceiling probabilities with sparsity param 
        self.probabilities -= np.min(self.probabilities)
        self.probabilities /= np.max(self.probabilities)
        self.probabilities -= self.sparsity

        # Generate logical target list based on cell probabilities
        self.targets = np.array(np.random.rand(len(self.probabilities)))
        self.targets = (self.targets <= self.probabilities)

        # Generate visited targets and visitation path list
        self.visited = [[] for item in range(len(self.targets))]
        self.path = []

    # Check current location of mouse against target list. Once target cell is visited, add to visited list.
    def check_location(self, m_loc):
        for ii in range(len(self.targets)):
            if self.cells[ii][0][0] < m_loc[0] < self.cells[ii][1][0]:
                if self.cells[ii][0][1] < m_loc[1] < self.cells[ii][1][1]:
                    if self.targets[ii] == 1:
                        self.targets[ii] = 0
                        cell_center_x = (self.cells[ii][0][0] + self.cells[ii][1][0]) // 2
                        cell_center_y = (self.cells[ii][0][1] + self.cells[ii][1][1]) // 2
                        self.visited[ii] = cell_center_x, cell_center_y
                        self.path.append([self.visited[ii], ii])

    def draw(self, canvas):
        for ii in range(len(self.cells)):
            target = self.targets[ii]
            cv2.rectangle(canvas, self.cells[ii][0], self.cells[ii][1], (int(255*target),0,0), thickness=-1)


"""
Gaussian maze and interactive visualization.
Target cells are marked as blue. Explore the maze with your mouse.
Once all targets are visited, a new maze is generated.
"""

# Generate maze and canvas
maze = gridMaze([1200,800], [12,8], std=20, sparsity=.3)
canvas = np.zeros([800,1200,3], dtype=np.uint8)
maze.draw(canvas)
realtime = True

# Start mouse callbacks
cv2.namedWindow('Gaussian Maze')
cv2.setMouseCallback('Gaussian Maze', mouse_loc)

# Draw Realtime updates
while realtime:

    # Check if mouse is inside a target rectangle
    maze.check_location(m_loc)

    # Mark visited targets
    cell_radius = maze.cellsize_x // 2
    for ii in range(len(maze.visited)):
                if np.sum(maze.visited[ii]) > 0:
                    cv2.rectangle(canvas, maze.cells[ii][0], maze.cells[ii][1], (0,0,0), thickness=-1)  
                    cv2.circle(canvas, maze.visited[ii], int(cell_radius*maze.probabilities[ii]), (0,0,255), thickness=-1)

    # Draw path
    if len(maze.path) >= 2:
        for ii in range(len(maze.path)-1):
            cv2.line(canvas, (maze.path[ii][0]), (maze.path[ii+1][0]), (255,255,255))
    
    # If targets have all been visited, generate new maze.
    if np.mean(maze.targets) == 0:
        maze = gridMaze([1200,800], [12,8], std=50, sparsity=.3)
        maze.draw(canvas)

    cv2.imshow("Gaussian Maze", canvas)
    if cv2.waitKey(1) == ord('q'):
        break