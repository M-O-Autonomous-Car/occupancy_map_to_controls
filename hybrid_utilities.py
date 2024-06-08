from constants import *
import serial
import math
import keyboard
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.spatial.kdtree as kd
from tqdm import tqdm


from classes import *

import HybridAstarPlanner.astar as astar
import HybridAstarPlanner.draw as draw
import CurvesGenerator.reeds_shepp as rs

# Function to create obstacles, by giving a list of x and y coordinates
def design_obstacles(x, y):
    ox, oy = [], []
    for i in range(x):
        ox.append(i)
        oy.append(0.0)
    for i in range(x):
        ox.append(x)
        oy.append(i)
    for i in range(x):
        ox.append(i)
        oy.append(y)
    for i in range(y):
        ox.append(0.0)
        oy.append(i)
    return ox, oy


def load_grid(file_name: str = "", x_max: int = MAP_SIZE_X_BLOCKS, y_max: int = MAP_SIZE_Y_BLOCKS):
    """
    Load a grid from a numpy array file

    Args:
        file_name (str): Name of the file to load

    Returns:
        np.ndarray: The loaded grid
    """
    if file_name == "":
        return np.zeros((y_max, x_max))
    # Check to see if file exists
    try:
        with open(f'maps/{file_name}.npy', 'r'):
            print(f"Loading {file_name}.npy")
            return np.load(f'maps/{file_name}.npy')
    except FileNotFoundError:
        print(f"File {file_name} not found")
        return np.zeros((y_max, x_max))

def draw_and_save(grid: np.ndarray):
    """
    Open a grid for the user, which they can draw on, and save the grid as a numpy array.
    """
    fig, ax = plt.subplots()
    drawing = False  # To track if the mouse button is pressed
    start_point = None  # To store the start point when drawing a rectangle

    def on_press(event):
        nonlocal drawing, start_point
        if event.button == 1:  # Left mouse button
            drawing = True
            if event.key == 'shift' and event.xdata is not None and event.ydata is not None:  # Start of rectangle draw
                start_point = (int(event.xdata), int(event.ydata))
            else:
                modify_grid(event)

    def on_release(event):
        nonlocal drawing, start_point
        if start_point:
            modify_rectangle(event)
        drawing = False
        start_point = None

    def on_move(event):
        if drawing:
            if start_point:
                modify_temporary_grid(event)  # Draw a rectangle to visualize the area
            else:
                modify_grid(event)  # Draw continuously

    def modify_grid(event):
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
                grid[y, x] = 1
                redraw_map()

    def modify_rectangle(event):
        end_point = (int(event.xdata), int(event.ydata))
        min_x, max_x = sorted([start_point[0], end_point[0]])
        min_y, max_y = sorted([start_point[1], end_point[1]])
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
                    grid[y, x] = 0 # Erase the rectangle
        redraw_map()

    def modify_temporary_grid(event):
        if event.xdata is not None and event.ydata is not None:
            ax.clear()
            redraw_map()
            end_x, end_y = int(event.xdata), int(event.ydata)
            rect_width, rect_height = abs(end_x - start_point[0]) + 1, abs(end_y - start_point[1]) + 1
            rect_x, rect_y = min(end_x, start_point[0]), min(end_y, start_point[1])
            ax.add_patch(patches.Rectangle((rect_x, rect_y), rect_width, rect_height, linewidth=1, edgecolor='r', facecolor='none'))
            plt.draw()

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

    def on_key(event):
        """
        Run A* search when 'a' is pressed and send the movements to the ESP32.

        Args:
            event (KeyEvent): Key press event

        Returns:
            None
        """
        if event.key == 'b':
            # Save the grid as a numpy array in the maps folder
            # And prompt the user for a name
            print("Saving the grid as a numpy array...")
            file_name = input("Enter a name for the map: ")
            np.save(f'maps/{file_name}.npy', grid)

        elif event.key == 'd':
            grid.fill(0)  # Clear the grid by filling it with zeros
            redraw_map()  # Redraw the map to reflect the cleared grid

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig.canvas.mpl_connect('key_press_event', on_key)
    redraw_map()
    plt.show()


def grid_to_obstacles(grid: np.ndarray):
    """
    Convert a grid to a list of obstacles

    Args:
        grid (np.ndarray): The grid to convert

    Returns:
        Tuple[np.ndarray, np.ndarray]: The x and y coordinates of the obstacles
    """
    ox, oy = np.array([]), np.array([])

    # Add in edges of the grid as obstacles
    for i in range(grid.shape[1]):
        ox = np.append(ox, i)
        oy = np.append(oy, 0)
    for i in range(grid.shape[1]):
        ox = np.append(ox, i)
        oy = np.append(oy, grid.shape[0] - 1)
    for i in range(grid.shape[0]):
        ox = np.append(ox, 0)
        oy = np.append(oy, i)
    for i in range(grid.shape[0]):
        ox = np.append(ox, grid.shape[1] - 1)
        oy = np.append(oy, i)

    obstacles = np.argwhere(grid == 1)
    ox = np.append(ox, obstacles[:, 1])
    oy = np.append(oy, obstacles[:, 0])

    return ox, oy


def hybrid_astar_planning(sx, sy, syaw, gx, gy, gyaw, ox, oy, xyreso, yawreso):
    sxr, syr = round(sx / xyreso), round(sy / xyreso)
    gxr, gyr = round(gx / xyreso), round(gy / xyreso)
    syawr = round(rs.pi_2_pi(syaw) / yawreso)
    gyawr = round(rs.pi_2_pi(gyaw) / yawreso)

    nstart = Node(sxr, syr, syawr, 1, [sx], [sy], [syaw], [1], 0.0, 0.0, -1)
    ngoal = Node(gxr, gyr, gyawr, 1, [gx], [gy], [gyaw], [1], 0.0, 0.0, -1)

    kdtree = kd.KDTree([[x, y] for x, y in zip(ox, oy)])
    P = calc_parameters(ox, oy, xyreso, yawreso, kdtree)

    print("[PLANNING] Calculating heuristic...")
    hmap = astar.calc_holonomic_heuristic_with_obstacle(ngoal, P.ox, P.oy, P.xyreso, 1.0)
    print("[PLANNING] Heuristic calculated...")
    steer_set, direc_set = calc_motion_set()
    open_set, closed_set = {calc_index(nstart, P): nstart}, {}

    qp = QueuePrior()
    qp.put(calc_index(nstart, P), calc_hybrid_cost(nstart, hmap, P))
    
    i = 0
    while True:
        i += 1
        # Every 50 iterations, print the number of iterations
        if i % 50 == 0:
            print(f"[PLANNING] Hybrid Planning Iteration: {i}")
        if not open_set:
            return None

        ind = qp.get()
        n_curr = open_set[ind]
        closed_set[ind] = n_curr
        open_set.pop(ind)

        update, fpath = update_node_with_analystic_expantion(n_curr, ngoal, P)

        if update:
            fnode = fpath
            break

        for i in range(len(steer_set)):
            node = calc_next_node(n_curr, ind, steer_set[i], direc_set[i], P)

            if not node:
                continue

            node_ind = calc_index(node, P)

            if node_ind in closed_set:
                continue

            if node_ind not in open_set:
                open_set[node_ind] = node
                qp.put(node_ind, calc_hybrid_cost(node, hmap, P))
            else:
                if open_set[node_ind].cost > node.cost:
                    open_set[node_ind] = node
                    qp.put(node_ind, calc_hybrid_cost(node, hmap, P))

    print("Finished planning, and found path...")
    return extract_path(closed_set, fnode, nstart)


def extract_path(closed, ngoal, nstart):
    rx, ry, ryaw, direc = [], [], [], []
    cost = 0.0
    node = ngoal

    while True:
        rx += node.x[::-1]
        ry += node.y[::-1]
        ryaw += node.yaw[::-1]
        direc += node.directions[::-1]
        cost += node.cost

        if is_same_grid(node, nstart):
            break

        node = closed[node.pind]

    rx = rx[::-1]
    ry = ry[::-1]
    ryaw = ryaw[::-1]
    direc = direc[::-1]

    direc[0] = direc[1]
    path = Path(rx, ry, ryaw, direc, cost)

    return path


def calc_next_node(n_curr, c_id, u, d, P):
    step = C.XY_RESO * 2

    nlist = math.ceil(step / C.MOVE_STEP)
    xlist = [n_curr.x[-1] + d * C.MOVE_STEP * math.cos(n_curr.yaw[-1])]
    ylist = [n_curr.y[-1] + d * C.MOVE_STEP * math.sin(n_curr.yaw[-1])]
    yawlist = [rs.pi_2_pi(n_curr.yaw[-1] + d * C.MOVE_STEP / C.WB * math.tan(u))]

    for i in range(nlist - 1):
        xlist.append(xlist[i] + d * C.MOVE_STEP * math.cos(yawlist[i]))
        ylist.append(ylist[i] + d * C.MOVE_STEP * math.sin(yawlist[i]))
        yawlist.append(rs.pi_2_pi(yawlist[i] + d * C.MOVE_STEP / C.WB * math.tan(u)))

    xind = round(xlist[-1] / P.xyreso)
    yind = round(ylist[-1] / P.xyreso)
    yawind = round(yawlist[-1] / P.yawreso)

    if not is_index_ok(xind, yind, xlist, ylist, yawlist, P):
        return None

    cost = 0.0

    if d > 0:
        direction = 1
        cost += abs(step)
    else:
        direction = -1
        cost += abs(step) * C.BACKWARD_COST

    if direction != n_curr.direction:  # switch back penalty
        cost += C.GEAR_COST

    cost += C.STEER_ANGLE_COST * abs(u)  # steer angle penalyty
    cost += C.STEER_CHANGE_COST * abs(n_curr.steer - u)  # steer change penalty
    cost = n_curr.cost + cost

    directions = [direction for _ in range(len(xlist))]

    node = Node(xind, yind, yawind, direction, xlist, ylist,
                yawlist, directions, u, cost, c_id)

    return node


def is_index_ok(xind, yind, xlist, ylist, yawlist, P):
    if xind <= P.minx or \
            xind >= P.maxx or \
            yind <= P.miny or \
            yind >= P.maxy:
        return False

    ind = range(0, len(xlist), C.COLLISION_CHECK_STEP)

    nodex = [xlist[k] for k in ind]
    nodey = [ylist[k] for k in ind]
    nodeyaw = [yawlist[k] for k in ind]

    if is_collision(nodex, nodey, nodeyaw, P):
        return False

    return True


def update_node_with_analystic_expantion(n_curr, ngoal, P):
    path = analystic_expantion(n_curr, ngoal, P)  # rs path: n -> ngoal

    if not path:
        return False, None

    fx = path.x[1:-1]
    fy = path.y[1:-1]
    fyaw = path.yaw[1:-1]
    fd = path.directions[1:-1]

    fcost = n_curr.cost + calc_rs_path_cost(path)
    fpind = calc_index(n_curr, P)
    fsteer = 0.0

    fpath = Node(n_curr.xind, n_curr.yind, n_curr.yawind, n_curr.direction,
                 fx, fy, fyaw, fd, fsteer, fcost, fpind)

    return True, fpath


def analystic_expantion(node, ngoal, P):
    sx, sy, syaw = node.x[-1], node.y[-1], node.yaw[-1]
    gx, gy, gyaw = ngoal.x[-1], ngoal.y[-1], ngoal.yaw[-1]

    maxc = math.tan(C.MAX_STEER) / C.WB
    paths = rs.calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=C.MOVE_STEP)

    if not paths:
        return None

    pq = QueuePrior()
    for path in paths:
        pq.put(path, calc_rs_path_cost(path))

    while not pq.empty():
        path = pq.get()
        ind = range(0, len(path.x), C.COLLISION_CHECK_STEP)

        pathx = [path.x[k] for k in ind]
        pathy = [path.y[k] for k in ind]
        pathyaw = [path.yaw[k] for k in ind]

        if not is_collision(pathx, pathy, pathyaw, P):
            return path

    return None


def is_collision(x, y, yaw, P):
    for ix, iy, iyaw in zip(x, y, yaw):
        d = 1
        dl = (C.RF - C.RB) / 2.0
        r = (C.RF + C.RB) / 2.0 + d

        cx = ix + dl * math.cos(iyaw)
        cy = iy + dl * math.sin(iyaw)

        ids = P.kdtree.query_ball_point([cx, cy], r)

        if not ids:
            continue

        for i in ids:
            xo = P.ox[i] - cx
            yo = P.oy[i] - cy
            dx = xo * math.cos(iyaw) + yo * math.sin(iyaw)
            dy = -xo * math.sin(iyaw) + yo * math.cos(iyaw)

            if abs(dx) < r and abs(dy) < C.W / 2 + d:
                return True

    return False


def calc_rs_path_cost(rspath):
    cost = 0.0

    for lr in rspath.lengths:
        if lr >= 0:
            cost += 1
        else:
            cost += abs(lr) * C.BACKWARD_COST

    for i in range(len(rspath.lengths) - 1):
        if rspath.lengths[i] * rspath.lengths[i + 1] < 0.0:
            cost += C.GEAR_COST

    for ctype in rspath.ctypes:
        if ctype != "S":
            cost += C.STEER_ANGLE_COST * abs(C.MAX_STEER)

    nctypes = len(rspath.ctypes)
    ulist = [0.0 for _ in range(nctypes)]

    for i in range(nctypes):
        if rspath.ctypes[i] == "R":
            ulist[i] = -C.MAX_STEER
        elif rspath.ctypes[i] == "WB":
            ulist[i] = C.MAX_STEER

    for i in range(nctypes - 1):
        cost += C.STEER_CHANGE_COST * abs(ulist[i + 1] - ulist[i])

    return cost


def calc_hybrid_cost(node, hmap, P):
    cost = node.cost + \
           C.H_COST * hmap[node.xind - P.minx][node.yind - P.miny]

    return cost


def calc_motion_set():
    s = np.arange(C.MAX_STEER / C.N_STEER,
                  C.MAX_STEER, C.MAX_STEER / C.N_STEER)

    steer = list(s) + [0.0] + list(-s)
    direc = [1.0 for _ in range(len(steer))] + [-1.0 for _ in range(len(steer))]
    steer = steer + steer

    return steer, direc


def is_same_grid(node1, node2):
    if node1.xind != node2.xind or \
            node1.yind != node2.yind or \
            node1.yawind != node2.yawind:
        return False

    return True


def calc_index(node, P):
    ind = (node.yawind - P.minyaw) * P.xw * P.yw + \
          (node.yind - P.miny) * P.xw + \
          (node.xind - P.minx)

    return ind


def calc_parameters(ox, oy, xyreso, yawreso, kdtree):
    minx = round(min(ox) / xyreso)
    miny = round(min(oy) / xyreso)
    maxx = round(max(ox) / xyreso)
    maxy = round(max(oy) / xyreso)

    xw, yw = maxx - minx, maxy - miny

    minyaw = round(-C.PI / yawreso) - 1
    maxyaw = round(C.PI / yawreso)
    yaww = maxyaw - minyaw

    return Para(minx, miny, minyaw, maxx, maxy, maxyaw,
                xw, yw, yaww, xyreso, yawreso, ox, oy, kdtree)


# Function to draw the car on the map
def draw_car(x, y, yaw, steer, color='black'):
    """
    Draw the car on the map.
    """
    car = np.array([[-C.RB, -C.RB, C.RF, C.RF, -C.RB],
                    [C.W / 2, -C.W / 2, -C.W / 2, C.W / 2, C.W / 2]])

    wheel = np.array([[-C.TR, -C.TR, C.TR, C.TR, -C.TR],
                      [C.TW / 4, -C.TW / 4, -C.TW / 4, C.TW / 4, C.TW / 4]])

    rlWheel = wheel.copy()
    rrWheel = wheel.copy()
    frWheel = wheel.copy()
    flWheel = wheel.copy()

    Rot1 = np.array([[math.cos(yaw), -math.sin(yaw)],
                     [math.sin(yaw), math.cos(yaw)]])

    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    frWheel = np.dot(Rot2, frWheel)
    flWheel = np.dot(Rot2, flWheel)

    frWheel += np.array([[C.WB], [-C.WD / 2]])
    flWheel += np.array([[C.WB], [C.WD / 2]])
    rrWheel[1, :] -= C.WD / 2
    rlWheel[1, :] += C.WD / 2

    frWheel = np.dot(Rot1, frWheel)
    flWheel = np.dot(Rot1, flWheel)

    rrWheel = np.dot(Rot1, rrWheel)
    rlWheel = np.dot(Rot1, rlWheel)
    car = np.dot(Rot1, car)

    frWheel += np.array([[x], [y]])
    flWheel += np.array([[x], [y]])
    rrWheel += np.array([[x], [y]])
    rlWheel += np.array([[x], [y]])
    car += np.array([[x], [y]])

    plt.plot(car[0, :], car[1, :], color)
    plt.plot(frWheel[0, :], frWheel[1, :], color)
    plt.plot(rrWheel[0, :], rrWheel[1, :], color)
    plt.plot(flWheel[0, :], flWheel[1, :], color)
    plt.plot(rlWheel[0, :], rlWheel[1, :], color)
    draw.Arrow(x, y, yaw, C.WB * 0.8, color)

def send_controls_uart(path) -> None:
    steers = translate_to_controls(path)
    try:
        with serial.Serial(USB_PORT, USB_BAUD_RATE, timeout=5) as ser:  # Increased timeout
            for angle in steers:
                # Stop the car if the loop flag is set to False
                if not loop_flag:
                    ser.write(b"stop\n")
                    print("Sent stop command to the ESP32.")
                    break

                command = f"{angle:.2f},{FORWARD_INTERVAL:.2f}\n"
                ser.write(command.encode('utf-8'))  # Send command as bytes
                print(f"Sent: {command.strip()}")

                # Wait for ESP32 to send back an acknowledgment over UART
                response = ser.readline().decode('utf-8').strip()
                print(f"Received: {response}")

        print(f"Sent commands to the ESP32.")
    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def send_one_control_uart(steer_rad: float) -> None:
    # Turn radians into degrees
    deg_delta = math.degrees(steer_rad)
    if deg_delta < 0:
        deg_angle = math.degrees(steer_rad) - ANGLE_MULTIPLIER + SERVO_DEFAULT_ANGLE
    else:
        deg_angle = math.degrees(steer_rad) + ANGLE_MULTIPLIER + SERVO_DEFAULT_ANGLE

    if abs(deg_delta) < 10:
        deg_delta *= 1.5

    forward_time = TURN_INTERVAL if abs(deg_delta) > TURN_THRESHOLD else FORWARD_INTERVAL

    try:
        with serial.Serial(USB_PORT, USB_BAUD_RATE, timeout=5) as ser:  # Increased timeout
            command = f"{deg_angle:.2f},{forward_time:.2f}\n"
            ser.write(command.encode('utf-8'))  # Send command as bytes
            print(f"Sent: {command.strip()}")

            # Wait for ESP32 to send back an acknowledgment over UART
            response = ser.readline().decode('utf-8').strip()
            print(f"Received: {response}")

        print(f"Sent commands to the ESP32.")
    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


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
    # Move for a certain amount of time given the angle, the distance do not matter
    # The distance will be constant
    prev_angle = 0
    relative_angles = []
    for i in tqdm(range(len(path))):
        angle = path.yaw[i]
        relative_angles.append(angle - prev_angle)
        prev_angle = angle

    return relative_angles
