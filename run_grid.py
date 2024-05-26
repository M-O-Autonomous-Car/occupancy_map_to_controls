from hybrid_utilities import *
from constants import *
import numpy as np
import argparse

def main(grid_name: str = "", grid_x_max: int = MAP_SIZE_X_BLOCKS, grid_y_max: int = MAP_SIZE_Y_BLOCKS) -> np.ndarray:
    grid = load_grid(grid_name, grid_x_max, grid_y_max)
    draw_and_save(grid)

if __name__ == "__main__":
    # Take in command line argument --show or --create
    parser = argparse.ArgumentParser(description="A more detailed example.")

    # Required positional argument
    parser.add_argument("--name", type=str, help="name of the grid")
    parser.add_argument("--x", type=int, help="x size of the grid")
    parser.add_argument("--y", type=int, help="y size of the grid")
    args = parser.parse_args()

    main(args.name, args.x, args.y)