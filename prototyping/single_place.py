import cv2
import numpy as np

"""
Functions
"""

# Generate target list (every grid cell in shuffled order)
def generate_targets(maze):
    targets =  np.random.permutation(len(maze.cells))
    return targets

# Get mouse location with cv2 mouse events
def mouse_loc(event, x, y, flags, param):
    global trial_state  # Declare that we will use the global trial_state
    if event == cv2.EVENT_MOUSEMOVE:
        m_loc[0] = x
        m_loc[1] = y
    if event == cv2.EVENT_LBUTTONDOWN:
        trial_state = True
        print(f"Next trial!")

# Mouse location/event monitor function. To be run every frame.
def check_loc(targets, m_loc_current, m_loc_prev, maze, trial_state=True):

    target = targets[1]
    found = False
    block_finished = False

    if len(targets) > 0 and trial_state == True:
        if maze.cells[target][0][0] < m_loc_current[0] < maze.cells[target][1][0]:
            if maze.cells[target][0][1] < m_loc_current[1] < maze.cells[target][1][1]:
                if m_loc_prev[0] <= m_loc_current[0] >= + m_loc_prev[0]:
                    if m_loc_prev[1] <= m_loc_current[1] >= + m_loc_prev[1]:
                        targets = targets[1:]  # Remove target if visited
                        found = True  # Record success (add timestamp)
                        trial_state = False  # Suspend trial state while mouse gets water
                        print('Target found! Wait for poke to begin next trial.')
                        return targets, found, block_finished, trial_state
    
    elif len(targets) == 0:
        print("Block complete!")
        block_finished = True

    return targets, found, block_finished, trial_state
         

"""
Draw equally spaced grid within bounds
"""

class gridMaze():
    def __init__(self, maze_bounds, maze_dims):
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
Start Code
"""
# Generate Maze
maze = gridMaze([1200,500], [12,5])

# Draw grid
def draw_grid(canvas):
    grid = np.zeros([canvas.bounds[1], canvas.bounds[0], 3], dtype=np.uint8)
    for ii in range(len(canvas.cells)):
        cv2.rectangle(grid, canvas.cells[ii][0], canvas.cells[ii][1], (255,255,255), thickness=2)
    return grid

# Initialize block
targets = generate_targets(maze)
block_finished = False
trial_state = True
m_loc = [1,1]
m_loc_prev = [1,1]
loc_history = [[1,1]]


print(m_loc[0])
print(f"target cell = {targets[1]}: {maze.cells[targets[1]][0][0]},{maze.cells[targets[1]][1][0]}")

# Start mouse callbacks
cv2.namedWindow('Clickbait')
cv2.setMouseCallback('Clickbait', mouse_loc)

"""
Update Code
"""

# Run block
while True:
    # Refresh arena visualization
    arena = draw_grid(maze)

    # Pull out a single cell for neighbor testing
    targets, found, block_finished, trial_state = check_loc(targets, m_loc, m_loc_prev, maze, trial_state)

    # Store previous mouse location
    loc_history.append(m_loc)

    # Set window of mouse location history
    loc_window = 10

    if len(loc_history) >= loc_window:
        loc_history = loc_history[:-1]

    # Calculate the mean of the previous {loc_window} mouse locations for x and y coordinates
    mean_x = int(np.mean([loc[0] for loc in loc_history]))
    mean_y = int(np.mean([loc[1] for loc in loc_history]))

    # Update m_loc_prev with the mean coordinates
    m_loc_prev = [mean_x, mean_y]
    
    #print(f"{m_loc}, {m_loc_prev}")


    cv2.rectangle(arena, maze.cells[targets[1]][0], maze.cells[targets[1]][1], (255,0,0), thickness=-1)  # Cell 8

    cv2.imshow("Clickbait", arena)
    if cv2.waitKey(16) == ord('q'):
        break