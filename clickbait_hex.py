import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

@dataclass
class HexTile:
    """Represents a single hexagonal tile in the grid."""
    index: Tuple[int, int]  # axial coordinates (q, r)
    center: Tuple[float, float]  # pixel coordinates (x, y)
    
class HexGrid:
    def __init__(self, center: Tuple[int, int], radius: int, hex_size: int):
        """
        Initialize a hexagonal grid that covers a circular area.
        
        Args:
            center: The center point of the circular area (x, y)
            radius: The radius of the circular area to cover
            hex_size: The size (radius) of each hexagon
        """
        self.center = center
        self.radius = radius
        self.hex_size = hex_size
        self.tiles: Dict[Tuple[int, int], HexTile] = {}
        self._initialize_grid()
        
    def _initialize_grid(self):
        """Calculate and store all hexagons needed to cover the circular area."""
        # Constants for hexagon spacing
        width = self.hex_size * 2
        vert_spacing = self.hex_size * np.sqrt(3)
        horiz_spacing = width * 3/4
        
        # Calculate the number of hexagons needed to cover the area
        max_q = int(np.ceil(self.radius / horiz_spacing))
        max_r = int(np.ceil(self.radius / vert_spacing))
        
        # Generate hexagons in axial coordinates
        for q in range(-max_q, max_q + 1):
            for r in range(-max_r, max_r + 1):
                # Convert axial coordinates to pixel coordinates
                x = self.center[0] + q * horiz_spacing
                y = self.center[1] + (r * vert_spacing + (q % 2) * vert_spacing/2)
                
                # Check if this hexagon's center is within the circular area
                if self._distance_from_center((x, y)) <= self.radius:
                    self.tiles[(q, r)] = HexTile((q, r), (x, y))
    
    def _distance_from_center(self, point: Tuple[float, float]) -> float:
        """Calculate distance between a point and the grid center."""
        return np.sqrt((point[0] - self.center[0])**2 + 
                      (point[1] - self.center[1])**2)
    
    def get_containing_hex(self, point: Tuple[float, float]) -> Optional[HexTile]:
        """
        Find the hexagon containing the given point.
        
        Args:
            point: The point to check (x, y)
            
        Returns:
            HexTile if point is inside a hexagon, None otherwise
        """
        # Quick check if point is even in the circular area
        if self._distance_from_center(point) > self.radius:
            return None
            
        # Convert point to axial coordinates (approximate)
        width = self.hex_size * 2
        vert_spacing = self.hex_size * np.sqrt(3)
        horiz_spacing = width * 3/4
        
        # Find the nearest q coordinate
        q = round((point[0] - self.center[0]) / horiz_spacing)
        
        # Adjust for the offset rows
        y_offset = (q % 2) * vert_spacing/2
        r = round((point[1] - self.center[1] - y_offset) / vert_spacing)
        
        # Get the candidate hex and its neighbors
        candidates = [(q, r)]
        for dq, dr in [(0,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0)]:
            candidates.append((q + dq, r + dr))
            
        # Check each candidate hex
        for q_r in candidates:
            if q_r in self.tiles:
                hex_tile = self.tiles[q_r]
                if point_in_hexagon(point, hex_tile.center, self.hex_size):
                    return hex_tile
        
        return None
    
    def draw(self, image: np.ndarray, 
            color: Tuple[int, int, int] = (64, 64, 64), 
            thickness: int = 1) -> np.ndarray:
        """Draw the entire hexagonal grid."""
        for tile in self.tiles.values():
            image = draw_hexagon(image, tile.center, self.hex_size, color, thickness)
        return image

def create_hexagon_points(center: Tuple[float, float], radius: int) -> np.ndarray:
    """Generate the six vertices of a regular hexagon."""
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack((x, y)).astype(np.int32)

def draw_hexagon(image: np.ndarray, 
                center: Tuple[float, float], 
                radius: int, 
                color: Tuple[int, int, int] = (0, 255, 0), 
                thickness: int = 1) -> np.ndarray:
    """Draw a single hexagon."""
    points = create_hexagon_points(center, radius)
    return cv2.polylines(image, [points], True, color, thickness, lineType=cv2.LINE_AA)

def point_in_hexagon(point: Tuple[float, float], 
                    hex_center: Tuple[float, float], 
                    radius: int) -> bool:
    """Check if a point lies inside a regular hexagon."""
    hex_points = create_hexagon_points(hex_center, radius)
    path = hex_points.reshape((-1, 1, 2))
    return cv2.pointPolygonTest(path, point, False) >= 0


def check_centroid_hex(cx, cy, param):
    img = param['base_img'].copy()
    hex_tile = param['grid'].get_containing_hex((cx, cy))
    
    if hex_tile:
        # Create a mask for the hexagon
        overlay = img.copy()
        # Draw filled hexagon contain the centroid
        points = create_hexagon_points(hex_tile.center, param['grid'].hex_size)
        cv2.fillPoly(overlay, [points], (128, 128, 255), lineType=cv2.LINE_AA)
        # Blend the overlay with the original image.
        opacity = 0.25
        cv2.addWeighted(overlay, opacity, img, 1-opacity, 0, img)
    
    return img


# Load cv2 video capture
video = cv2.VideoCapture('A:/neurodinner/skymouse_cropped.mp4')
thresh = cv2.VideoCapture('A:/neurodinner/skymouse_cropped_threshold.mp4')

while True:
    ret, frame = video.read()
    ret_thresh, frame_thresh = thresh.read()
    if not ret:
        break

    
    # Convert threshold frame to binary
    frame_thresh = cv2.cvtColor(frame_thresh, cv2.COLOR_BGR2GRAY)
    _, frame_thresh = cv2.threshold(frame_thresh, 128, 255, cv2.THRESH_BINARY)
    # Invert binary image
    frame_thresh = cv2.bitwise_not(frame_thresh)
    # Expand binary image
    kernel = np.ones((5, 5), np.uint8)
    frame_thresh = cv2.dilate(frame_thresh, kernel, iterations=2)
    # Erode binary image
    frame_thresh = cv2.erode(frame_thresh, kernel, iterations=3)
    
    # Get largest connected component and find centroid
    contours, _ = cv2.findContours(frame_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # Draw centroid
    frame = cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    # Create a window and blank image
    width, height = frame.shape[0], frame.shape[1]

    # Create a hex grid centered in the window
    grid = HexGrid(center=(width//2, height//2), radius=450, hex_size=50)

    # Draw the hexagon containing the centroid
    frame = check_centroid_hex(cx, cy, {'grid': grid, 'base_img': frame})

    # Draw the base grid
    base_img = grid.draw(frame.copy())



    cv2.imshow('Hex Grid', base_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
            
