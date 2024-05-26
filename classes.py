import math
import numpy as np
from heapdict import heapdict

from constants import *

class C:  # Parameter config
    PI = math.pi  # 3.141592653589793

    # Constants with placeholders for BLOCK_PER_METER and DEGREE_RESOLUTION
    XY_RESO = BLOCK_PER_METER  # [m] (Default BLOCK_PER_METER e.g., 1.0)
    YAW_RESO = np.deg2rad(DEGREE_RESOLUTION)  # [rad] (Default np.deg2rad(DEGREE_RESOLUTION) e.g., np.deg2rad(15))
    MOVE_STEP = MOVE_REOLUTION  # [m] path interpolate resolution (0.4)
    N_STEER = 20.0  # steer command number (20.0)
    COLLISION_CHECK_STEP = 5  # skip number for collision check (5)
    EXTEND_BOUND = 1  # collision check range extended (1)

    GEAR_COST = 100.0  # switch back penalty cost (100.0)
    BACKWARD_COST = 5.0  # backward penalty cost (5.0)
    STEER_CHANGE_COST = 10.0  # steer angle change penalty cost (5.0)
    STEER_ANGLE_COST = 1.0  # steer angle penalty cost (1.0)
    H_COST = 15.0  # Heuristic cost penalty cost (15.0)

    # Constants with scaling applied; assuming initial constants are defined elsewhere
    RF = RF * ALGO_SCALING  # [m] distance from rear to vehicle front end of vehicle (Assuming initial RF e.g., 4.5)
    RB = RB * ALGO_SCALING  # [m] distance from rear to vehicle back end of vehicle (Assuming initial RB e.g., 1.0)
    W = W * ALGO_SCALING  # [m] width of vehicle (Assuming initial W e.g., 3.0)
    WD = WD * ALGO_SCALING  # [m] distance between left-right wheels (Assuming initial WD e.g., 0.7 * W)
    WB = WB * ALGO_SCALING  # [m] Wheel base (Assuming initial WB e.g., 3.5)
    TR = TR * ALGO_SCALING  # [m] Tyre radius (Assuming initial TR e.g., 0.5)
    TW = TW * ALGO_SCALING  # [m] Tyre width (Assuming initial TW e.g., 1.0)
    MAX_STEER = MAX_STEER  # [rad] maximum steering angle (Assuming initial MAX_STEER e.g., 0.6)

class Node:
    def __init__(self, xind, yind, yawind, direction, x, y,
                 yaw, directions, steer, cost, pind):
        self.xind = xind
        self.yind = yind
        self.yawind = yawind
        self.direction = direction
        self.x = x
        self.y = y
        self.yaw = yaw
        self.directions = directions
        self.steer = steer
        self.cost = cost
        self.pind = pind


class Para:
    def __init__(self, minx, miny, minyaw, maxx, maxy, maxyaw,
                 xw, yw, yaww, xyreso, yawreso, ox, oy, kdtree):
        self.minx = minx
        self.miny = miny
        self.minyaw = minyaw
        self.maxx = maxx
        self.maxy = maxy
        self.maxyaw = maxyaw
        self.xw = xw
        self.yw = yw
        self.yaww = yaww
        self.xyreso = xyreso
        self.yawreso = yawreso
        self.ox = ox
        self.oy = oy
        self.kdtree = kdtree


class Path:
    def __init__(self, x, y, yaw, direction, cost):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.direction = direction
        self.cost = cost


class QueuePrior:
    def __init__(self):
        self.queue = heapdict()

    def empty(self):
        return len(self.queue) == 0  # if Q is empty

    def put(self, item, priority):
        self.queue[item] = priority  # push 

    def get(self):
        return self.queue.popitem()[0]  # pop out element with smallest priority