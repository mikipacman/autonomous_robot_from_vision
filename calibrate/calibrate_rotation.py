import time

import cv2
import numpy as np
from car_lib.cars import RESOLUTIONS, Camera, Connection, Direction, Motors
from car_lib.my_helpers import (detect_aruco, get_calibration_parameters,
                                get_one_marker_vecs,
                                get_one_marker_vecs_until_success,
                                rvec_to_angle)

# Program params
power = 62
time_to_rotate = 0.5


def main():
    # init basic connection etc
    cv2.namedWindow("original")
    connection = Connection()
    cam = Camera(connection=connection)
    motors = Motors(connection=connection, motor2_multiplier=0.99)
    w, h = RESOLUTIONS[cam.get_quality()]
    assert w == 640 and h == 480
    folder_path = "images_for_calibration/640x480"

    # init camera params
    params = get_calibration_parameters(folder_path)
    ret, mtx, dist, rvecs, tvecs, mapx, mapy, new_camera_mat, roi = params

    # Get initial position
    rvec, _ = get_one_marker_vecs_until_success(cam, params)
    start_time = time.time()
    start_angle = rvec_to_angle(rvec)

    # drive for a given time
    while time.time() - start_time < time_to_rotate:
        connection.keep_stream_alive()
        img = cam.get_frame()

        # drive forward
        motors.command(power, Direction.LEFT)

        # show im
        cv2.imshow("original", img)
        cv2.pollKey()

        ret = get_one_marker_vecs(img, params)
        if ret is not None:
            rvec, _ = ret
            print("current angle:", rvec_to_angle(rvec))

    # Get final position
    rvec, _ = get_one_marker_vecs_until_success(cam, params)
    end_time = time.time()
    end_angle = rvec_to_angle(rvec)

    pos_diff = start_angle - end_angle
    time_diff = end_time - start_time

    print("velocity of robot in degrees/second =", pos_diff / time_diff)


if __name__ == "__main__":
    main()
