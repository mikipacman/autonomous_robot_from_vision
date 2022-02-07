import time

import cv2
import numpy as np

from car_lib.cars import Direction
from car_lib.robot import Robot


def main():
    marker_to_dir = {
        0: Direction.FORWARD,
        4: Direction.BACKWARD,
        6: Direction.LEFT,
        8: Direction.RIGHT,
    }

    robot = Robot(flash=True)

    while True:
        img = robot._get_image()
        ret = robot._get_marker_from_image(img)
        if ret is not None:
            ids = ret[1]
            assert len(ids) == 1
            dir = marker_to_dir[int(ids[0])]
            robot.motors.command(65, dir)


if __name__ == "__main__":
    main()
