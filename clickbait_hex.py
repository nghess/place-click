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
            color: Tuple[int, int, int] = (0, 255, 0), 
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
    return cv2.polylines(image, [points], True, color, thickness)

def point_in_hexagon(point: Tuple[float, float], 
                    hex_center: Tuple[float, float], 
                    radius: int) -> bool:
    """Check if a point lies inside a regular hexagon."""
    hex_points = create_hexagon_points(hex_center, radius)
    path = hex_points.reshape((-1, 1, 2))
    return cv2.pointPolygonTest(path, point, False) >= 0

# Example usage with mouse interaction
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        img = param['base_img'].copy()
        hex_tile = param['grid'].get_containing_hex((x, y))
        
        if hex_tile:
            # Highlight the hexagon containing the mouse
            draw_hexagon(img, hex_tile.center, param['grid'].hex_size, 
                        (0, 0, 255), 2)
            # Display the hexagon's index
            cv2.putText(img, f"Hex {hex_tile.index}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('Hex Grid', img)

if __name__ == "__main__":
    # Create a window and blank image
    width, height = 800, 600
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a hex grid centered in the window
    grid = HexGrid(center=(width//2, height//2), radius=200, hex_size=30)
    
    # Draw the base grid
    base_img = grid.draw(img.copy())
    
    # Set up mouse callback
    cv2.namedWindow('Hex Grid')
    cv2.setMouseCallback('Hex Grid', mouse_callback, 
                        {'grid': grid, 'base_img': base_img})
    
    # Main loop
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()