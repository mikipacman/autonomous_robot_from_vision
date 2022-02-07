import time

import cv2
import numpy as np

from car_lib.cars import Direction
from car_lib.robot import Robot


def get_linear_step_func(p1, p2):
    def f(x_input):
        x = (p1[0], p2[0])
        y = (p1[1], p2[1])
        poly = np.poly1d(np.polyfit(x, y, 1))
        return np.clip(poly(x_input), min(y), max(y))

    return f


def angle_margin_func(dist, adjusted_course):
    if adjusted_course:
        return 25
    else:
        return get_linear_step_func((1, 25), (2, 15))(dist)


def get_maneuver_angle_and_dist(angle, dist):
    get_a = get_linear_step_func((3, 0.4), (1.5, 1))
    get_b = get_linear_step_func((1.5, 1.0), (2.5, 0.5))

    sign = -np.sign(angle)
    angle_deg = np.abs(angle)
    angle = np.abs(angle) / 180 * 3.14
    a = get_a(dist)
    b = get_b(angle_deg)
    z = dist * np.cos(angle)
    y = dist * np.sin(angle)
    gamma_deg = np.arctan((1 - a) * z / y) / 3.14 * 180
    man_angle = 90 - angle_deg - gamma_deg
    man_dist = np.sqrt(y ** 2 + ((1 - a) * z) ** 2)
    return sign * man_angle, man_dist * b


def go_to_marker(robot, marker_id):
    dir = Direction.LEFT
    adjusted_course = False
    while True:
        angle, dist = robot.find_marker(marker_id, start_dir=dir)
        if dist < 0.7:
            break

        if np.abs(angle) < angle_margin_func(dist, adjusted_course) or dist < 1:
            robot.drive_straight()
        else:
            man_angle, man_dist = get_maneuver_angle_and_dist(angle, dist)

            robot.rotate_approximately(man_angle)
            robot.drive_straight_appoximately(man_dist)
            robot.rotate_approximately(-man_angle)

            dir = Direction.RIGHT if angle > 0 else Direction.LEFT
            adjusted_course = True


def main():
    # marker_ids = [0, 6, 5, 4] # Circle
    marker_ids = [0, 5, 4, 6]  # X

    robot = Robot()

    for marker_id in marker_ids:
        go_to_marker(robot, marker_id)
        print("I found marker", marker_id, "!")
        robot.blink()


if __name__ == "__main__":
    main()
