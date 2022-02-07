#!/usr/bin/env python3
import time

import cv2
from car_lib.cars import RESOLUTIONS, Camera, Connection, Direction, Motors

help = """
=====[ HELP ]=====
q - quit
h - help

Image:
= - bigger res
- - lower res
+ - bigger picture on screen 
_ - smaller scaling factor
f - toggle flash

Steering:
a - left
d - right
w - forward
s - backward

] - bigger power
[ - smaller power
"""


min_power, max_power = 62, 100


def main():
    cv2.namedWindow("demo")
    connection = Connection()
    cam = Camera(connection=connection)
    motors = Motors(connection=connection, motor2_multiplier=0.99)

    power, turning_power, flash, scaling = 80, 65, False, 2

    while True:
        connection.keep_stream_alive()

        img = cam.get_frame()
        dim = (img.shape[1] * scaling, img.shape[0] * scaling)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow("demo", img)

        keypress = cv2.pollKey() & 0xFF
        if keypress == ord("q"):
            break
        elif keypress == ord("h"):
            print(help)

        elif keypress == ord("="):
            q = cam.get_quality()
            if not q == max(RESOLUTIONS.keys()):
                cam.set_quality(q + 1)
        elif keypress == ord("-"):
            q = cam.get_quality()
            if not q == min(RESOLUTIONS.keys()):
                cam.set_quality(q - 1)
        elif keypress == ord("+"):
            scaling += 1
        elif keypress == ord("_"):
            scaling = max(1, scaling - 1)
        elif keypress == ord("f"):
            if not flash:
                cam.flash_on()
            else:
                cam.flash_off()
            flash = not flash

        elif keypress == ord("a"):
            motors.command(turning_power, Direction.LEFT)
        elif keypress == ord("d"):
            motors.command(turning_power, Direction.RIGHT)
        elif keypress == ord("w"):
            motors.command(power, Direction.FORWARD)
        elif keypress == ord("s"):
            motors.command(power, Direction.BACKWARD)

        elif keypress == ord("]"):
            power = min(power + 1, max_power)
            print("current power", power)
        elif keypress == ord("["):
            power = max(power - 1, min_power)
            print("current power", power)


if __name__ == "__main__":
    main()
