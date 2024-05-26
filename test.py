import math
# Simple function for converting radians to degrees
def radians_to_degrees(radians):
    return radians * 180 / math.pi

def degrees_to_radians(degrees):
    return degrees * math.pi / 180


print(radians_to_degrees(0.6))
print(degrees_to_radians(45))