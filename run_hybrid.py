# Main function for the o2c project

from hybrid_utilities import *
from constants import *

import os
import sys
import math
import argparse
from heapdict import heapdict
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.kdtree as kd

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../MotionPlanning/")

import CurvesGenerator.reeds_shepp as rs

HPS = (5, 2) # Start position of the path
HPG = (20, 18) # Goal position of the path

def design_obstacles(x, y):
    ox, oy = [], []

    for i in range(x):
        ox.append(i)
        oy.append(0)
    for i in range(x):
        ox.append(i)
        oy.append(y - 1)
    for i in range(y):
        ox.append(0)
        oy.append(i)
    for i in range(y):
        ox.append(x - 1)
        oy.append(i)
    for i in range(10, 21):
        ox.append(i)
        oy.append(15)
    for i in range(15):
        ox.append(20)
        oy.append(i)
    for i in range(15, 30):
        ox.append(30)
        oy.append(i)
    for i in range(16):
        ox.append(40)
        oy.append(i)

    return ox, oy

def main_a_star(grid_name="grid3"):
    print("start!")
    grid = load_grid(grid_name)
    ox, oy = grid_to_obstacles(grid)

    x, y = grid.shape
    sx, sy, syaw0 = PATH_START[0], PATH_START[1], np.deg2rad(ANGLE_START)
    gx, gy, gyaw0 = PATH_GOAL[0], PATH_GOAL[1], np.deg2rad(ANGLE_GOAL)

    t0 = time.time()
    print("Planning ...")
    path = hybrid_astar_planning(sx, sy, syaw0, gx, gy, gyaw0,
                                 ox, oy, C.XY_RESO, C.YAW_RESO)
    t1 = time.time()
    print("running T: ", t1 - t0)

    if not path:
        print("Searching failed!")
        return
    print("##############################")
    print("Path found!")
    print("##############################")

    # Print the path first
    plt.plot(ox, oy, "sk")
    plt.plot(path.x, path.y, linewidth=1.5, color='r')
    plt.title("Hybrid A*")
    plt.axis("equal")
    plt.show()

    user_input = input("Press Enter to continue, q to quit:")
    if user_input == "q":
        return

    x = path.x
    y = path.y
    yaw = path.yaw
    direction = path.direction

    for k in range(len(x)):
        plt.cla()
        plt.plot(ox, oy, "sk")
        plt.plot(x, y, linewidth=1.5, color='r')

        steer = 0.0
        if k < len(x) - 2:
            dy = (yaw[k + 1] - yaw[k]) / C.MOVE_STEP
            steer = rs.pi_2_pi(math.atan(-C.WB * dy / direction[k]))

        draw_car(gx, gy, gyaw0, 0.0, 'dimgray')
        draw_car(x[k], y[k], yaw[k], steer)
        print("x: ", x[k], "y: ", y[k], "yaw: ", yaw[k], "steer: ", math.degrees(steer) * ANGLE_MULTIPLIER)
        print()
        plt.title("Hybrid A*")
        plt.axis("equal")
        plt.pause(0.0001)

        # input("Press Enter to continue...")

        # Send command to the car
        send_one_control_uart(steer)

    plt.show()
    print("Done!")

if __name__ == "__main__":
    # Take in command line argument --show or --create
    parser = argparse.ArgumentParser(description="A more detailed example.")

    # Required positional argument
    parser.add_argument("name", help="name of the grid", default="")
    args = parser.parse_args()

    print(design_obstacles(MAP_SIZE_X_BLOCKS, MAP_SIZE_Y_BLOCKS))

    # Hybrid A* algorithm
    main_a_star(args.name)
