import serial
import math
import keyboard
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from queue import PriorityQueue

from constants import *

def create_map():
    """ 
    Create an interactive occupancy map using matplotlib. 

    Returns:
        np.array: Grid map of the environment
    """
    fig, ax = plt.subplots()
    grid = np.zeros((MAP_SIZE_X_BLOCKS, MAP_SIZE_Y_BLOCKS))  # Define the size of the grid
    drawing = False  # State to check if mouse is pressed

    def on_press(event):
        nonlocal drawing
        drawing = True
        modify_grid(event)

    def on_release(event):
        nonlocal drawing
        drawing = False

    def on_move(event):
        if drawing:
            modify_grid(event)

    def modify_grid(event):
        # Convert click location to grid index
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0] and grid[y, x] == 0:
                grid[y, x] = 1  # Set the cell as blocked
                redraw_map()

    def on_key(event):
        """
        Run A* search when 'a' is pressed and send the movements to the ESP32.

        Args:
            event (KeyEvent): Key press event

        Returns:
            None
        """
        if event.key == 'a':
            path = a_star_search(grid, PATH_START, PATH_GOAL)
            if path:
                # Flip x and y to match the axis when plotting
                # Plot the first one as red, end as green, and the rest as blue
                for i, (y, x) in enumerate(path):
                    color = 'red' if i == 0 else 'green' if i == len(path) - 1 else 'blue'
                    ax.add_patch(patches.Rectangle((x, y), 1, 1, color=color, fill=True))
                plt.draw()
                movements = translate_to_controls(path)
                send_movements_via_uart(movements)
            else:
                print("No valid path found.")
        elif event.key == 'd':
            grid.fill(0)  # Clear the grid by filling it with zeros
            redraw_map()  # Redraw the map to reflect the cleared grid

    def redraw_map():
        ax.clear()
        ax.set_xlim(0, grid.shape[1])
        ax.set_ylim(0, grid.shape[0])
        ax.set_xticks(np.arange(0, grid.shape[1] + 1))
        ax.set_yticks(np.arange(0, grid.shape[0] + 1))
        ax.grid(True)
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if grid[y, x] == 1:
                    ax.add_patch(patches.Rectangle((x, y), 1, 1, color='black', fill=True))
        plt.draw()

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig.canvas.mpl_connect('key_press_event', on_key)
    redraw_map()
    plt.show()
    return grid

def heuristic(a, b):
    """Calculate the Manhattan distance heuristic for A*."""
    return abs(b[0] - a[0]) + abs(b[1] - a[1])

def is_valid_turn(grid, current, direction):
    """
    Check if a turn is valid based on the current position and direction for an Ackermann vehicle.

    For example, if the vehicle is moving in the top right direction and it is currently on (0,0), the following blocks should be valid:
    (0,0), (0,1), (1,0), (1,1), (1,2), (2,1), (2,2)
    """
    x, y = current
    dx, dy = direction
    # Determine squares to check based on direction
    if dx == -1 and dy == -1:  # top left
        turn_squares = [
            (x, y), (x-1, y), (x, y-1), (x-1, y-1),
            (x-2, y-1), (x-1, y-2), (x-2, y-2)
        ]
    elif dx == -1 and dy == 1:  # top right
        turn_squares = [
            (x, y), (x-1, y), (x, y+1), (x-1, y+1),
            (x-2, y+1), (x-1, y+2), (x-2, y+2)
        ]
    elif dx == 1 and dy == -1:  # bottom left
        turn_squares = [
            (x, y), (x+1, y), (x, y-1), (x+1, y-1),
            (x+2, y-1), (x+1, y-2), (x+2, y-2)
        ]
    elif dx == 1 and dy == 1:  # bottom right
        turn_squares = [
            (x, y), (x+1, y), (x, y+1), (x+1, y+1),
            (x+2, y+1), (x+1, y+2), (x+2, y+2)
        ]
    else:  # For orthogonal movements, we check a 1x1 space
        turn_squares = [(x + dx, y + dy)]
    
    for sx, sy in turn_squares:
        if not (0 <= sx < grid.shape[0] and 0 <= sy < grid.shape[1]):
            return False
        if grid[sx, sy] != 0:
            return False
    return True

def a_star_search(grid, start, goal):
    """
    Perform A* search on the grid.

    Note:
    When there are turns, it has to be 2 blocks away and two blocks in the turning direction for a turn to work

    Args:
        grid (np.array): Map of the environment
        start (tuple): Start position
        goal (tuple): Goal position

    Returns:
        list: Path from start to goal
    """
    # Switch positions for start and end coords
    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while not frontier.empty():
        _, current = frontier.get()
        
        if current == goal:
            break
        
        # Allow movements forward and slight turns
        movements = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # 4 orthogonal movements
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # 4 diagonal movements
        ]
        
        for dx, dy in movements:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1] and grid[neighbor[0], neighbor[1]] == 0:
                if not is_valid_turn(grid, current, (dx, dy)):
                    continue
                
                # Cost for diagonal movement is sqrt(2), approximated as 1.414
                step_cost = 1.414 if dx != 0 and dy != 0 else 1
                new_cost = cost_so_far[current] + step_cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    frontier.put((priority, neighbor))
                    came_from[neighbor] = current

    return reconstruct_path(came_from, start, goal)

def reconstruct_path(came_from, start, goal):
    """ Reconstruct the path from start to goal using the came_from map. """
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from.get(current, start)
    path.append(start)
    path.reverse()
    return path if len(path) > 2 else None # Exclude start and goal if they are the only points

def translate_to_controls(path):
    """ 
    Translate path to control movements for an Ackermann vehicle. This function translates the path
    to a series of angle and speed changes, where each movement will take into account the vehicle's
    turning capabilities.

    Args:
        path (list): Path from start to goal represented as (x, y) tuples

    Returns:
        list: List of (angle, speed) tuples for each segment of the path
    """
    movements = []
    # Assume initial direction is facing 'up' (north)
    current_angle = 0  # degrees (north)
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        target_angle = math.degrees(math.atan2(dy, dx))  # Calculate target angle based on dy, dx

        print(f"From {path[i-1]} to {path[i]}: dx={dx}, dy={dy}, target_angle={target_angle}")
        
        # If angle is larger or smaller than the max and min angles, set it to the max or min angle
        angle_change = target_angle - current_angle  # Calculate change needed

        # Calculate distance to travel in meters and then the time needed at constant speed
        distance = math.sqrt(dx**2 + dy**2) * BLOCK_METERS  # Convert grid distance to meters

        # If the angle change is not zero, add the turning time to the forward time
        time_forward = distance / VEH_SPEED * (TURN_TIME_MULTI if angle_change != 0 else 1)
        
        movements.append((angle_change + SERVO_DEFAULT_ANGLE, time_forward))  # Append the tuple (angle change, time)
        current_angle = target_angle  # Update current angle to target angle for next calculation

    # Add final statement to stop car
    movements.append((SERVO_DEFAULT_ANGLE, 0))
    print("movements", movements)
    return movements

def stop_car(event):
    """
    Stop the car by sending a stop command to the ESP32 over UART if "s" is pressed.
    """
    global loop_flag
    if event.name == 's':
        loop_flag = False
        print("Stopping the car.")

def send_movements_via_uart(movements):
    """ Send movements to an ESP32 over UART, one by one. """
    try:
        with serial.Serial(USB_PORT, USB_BAUD_RATE, timeout=5) as ser:  # Increased timeout
            for angle, time_forward in movements:
                # Stop the car if the loop flag is set to False
                if not loop_flag:
                    ser.write(b"stop\n")
                    print("Sent stop command to the ESP32.")
                    break

                command = f"{angle:.2f},{time_forward:.2f}\n"
                ser.write(command.encode('utf-8'))  # Send command as bytes
                print(f"Sent: {command.strip()}")

                # Wait for ESP32 to send back an acknowledgment over UART
                response = ser.readline().decode('utf-8').strip()
                print(f"Received: {response}")
                
                # Check if the acknowledgment is valid (you may need to adapt this based on your ESP32's response format)

        print(f"Sent {len(movements)} movements to the ESP32.")
    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

