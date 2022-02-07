import time

import cv2
import numpy as np

from car_lib.cars import Direction
from car_lib.robot import Robot


def pick_better_basket(baskets_params):
    assert 2 >= len(baskets_params) > 0
    if len(baskets_params) == 1:
        return baskets_params[0]
    else:
        if baskets_params[0]["score"] > baskets_params[1]["score"]:
            return baskets_params[0]
        else:
            return baskets_params[1]


def align_basket(robot):
    while True:
        basket = pick_better_basket(robot.get_baskets_params())
        center = (basket["left"] + basket["right"]) / 2
        center_align = 0.57
        center_align_margin = 0.03

        if np.abs(center - center_align) < center_align_margin:
            pass
        elif center < center_align:
            robot.rotate(Direction.LEFT)
        else:
            robot.rotate(Direction.RIGHT)

        basket = pick_better_basket(robot.get_baskets_params())
        top = basket["top"]
        if top < 0.4:
            robot.drive_straight()
        else:
            return basket["color"]


def putdown_basket(robot):
    robot.servo_straight()
    robot.drive_straight_appoximately(0.5, Direction.BACKWARD)


def drive_to_place(robot, marker_id, pos):
    robot.find_marker(marker_id, Direction.LEFT)
    angle, dist = robot.drive_to_pos_in_marker_coord(marker_id, pos)
    robot.rotate_approximately(angle)
    robot.drive_straight_appoximately(dist)
    dir = Direction.RIGHT if angle < 0 else Direction.LEFT
    robot.find_marker(marker_id, dir)


def main():
    base_marker_ids = [7, 9]
    marker_to_pos = {
        7: (1, -0.1, 0),
        9: (0.2, -1, 0),
    }
    color_to_target_id = {"red": 6, "green": 8}
    target_to_pos = {
        6: (0, -1.3, 0),
        8: (0, -1.3, 0),
    }

    robot = Robot(flash=True)

    # Find base with basket
    found_marker_ids = robot.find_any_marker(base_marker_ids, Direction.LEFT)
    marker_id = found_marker_ids[0]
    pos = marker_to_pos[marker_id]

    print("Found marker", marker_id, "!")
    drive_to_place(robot, marker_id, pos)

    # pick up
    robot.servo_straight()
    color = align_basket(robot)
    robot.servo_pickup()

    target_marker_id = color_to_target_id[color]
    pos = target_to_pos[target_marker_id]

    print("Found", color, "basket !")
    drive_to_place(robot, target_marker_id, pos)

    # put down
    robot.drive_straight_appoximately(0.7)
    putdown_basket(robot)
    robot.blink()


if __name__ == "__main__":
    main()
