# Python file for defining constants
import math

# NAMES
USB_PORT = "/dev/cu.wchusbserial54E20370101" # USB port for the ESP32
USB_BAUD_RATE = 115200 # Baud rate for the ESP32
SERIAL_SEQ_NUM = 0 # Sequence number for the serial communication

# CONSTRAINTS
ALGO_SCALING = 6 # Scaling factor for the algorithm

METER_PER_BLOCK = 0.25 # Number of meters per block
BLOCK_PER_METER = 1/(METER_PER_BLOCK * ALGO_SCALING) # Number of blocks per meter
DEGREE_RESOLUTION = math.radians(1)  # Resolution, angles to radians
MAP_SIZE_X = 5 # Map size in the x direction in meters
MAP_SIZE_Y = 5 # Map size in the y direction in meters
MAP_SIZE_X_BLOCKS = int(math.ceil(MAP_SIZE_X * BLOCK_PER_METER)) # Map size in the x direction
MAP_SIZE_Y_BLOCKS = int(math.ceil(MAP_SIZE_Y * BLOCK_PER_METER)) # Map size in the y direction

PATH_START = (5, 2) # Start position of the path
PATH_GOAL = (20, 18) # Goal position of the path
PATH_GOAL2 = (5, 18) # Goal position of the path
ANGLE_START = 0 # Start angle of the path
ANGLE_GOAL = 90 # Goal angle of the path
ANGLE_GOAL2 = 0 # Goal angle of the path
MOVE_REOLUTION = 0.8 # Path interpolate resolution

BLOCK_METERS = 0.2 # 1 block = 0.1 meters
VEH_SPEED = 1 # Vehicle speed in m/s
TURN_TIME_MULTI = 0.8 # Multiplier for turning time
FORWARD_INTERVAL = 0.55 # Forward interval in seconds
TURN_INTERVAL = 0.8 # Turn interval in seconds
TURN_THRESHOLD = 5 # Turn threshold in degrees

MAX_ANGLE_RIGHT = 135 # Maximum angle to the right
MAX_ANGLE_LEFT = 45 # Maximum angle to the left
SERVO_DEFAULT_ANGLE = 90 # Default angle for the servo
ANGLE_MULTIPLIER = 3 # Multiplier for the angle

RF = 0.27  # [m] distance from rear to vehicle front end of vehicle
RB = 0  # [m] distance from rear to vehicle back end of vehicle
W = 0.197  # [m] width of vehicle
WD = 0.118 # [m] distance between left-right wheels
WB = 0.17  # [m] Wheel base (distance between front and rear wheel center)
TR = 0.0375  # [m] Tyre radius
TW = 0.03  # [m] Tyre width
MAX_STEER = 0.78539  # [rad] maximum steering angle
# MAX_STEER = 0.60  # [rad] maximum steering angle

loop_flag = True
