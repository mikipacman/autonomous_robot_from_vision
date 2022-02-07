import time

import cv2
import numpy as np
from car_lib.cars import RESOLUTIONS, Camera, Connection, Direction, Motors
from car_lib.my_helpers import (detect_aruco, get_calibration_parameters,
                                get_one_marker_vecs,
                                get_one_marker_vecs_until_success)

# Program params
power = 66
time_to_drive = 2


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
    _, tvec = get_one_marker_vecs_until_success(cam, params)
    start_time = time.time()
    start_pos = tvec[0][2]

    # drive for a given time
    while time.time() - start_time < time_to_drive:
        connection.keep_stream_alive()
        img = cam.get_frame()

        # drive forward
        motors.command(power, Direction.FORWARD)

        # show im
        cv2.imshow("original", img)
        cv2.pollKey()

        ret = get_one_marker_vecs(img, params)
        if ret is not None:
            _, tvec = ret
            print("distance from marker:", tvec[0][2])

    # Get final position
    _, tvec = get_one_marker_vecs_until_success(cam, params)
    end_time = time.time()
    end_pos = tvec[0][2]

    pos_diff = start_pos - end_pos
    time_diff = end_time - start_time

    print("velocity of robot in (unit of marker space)/second =", pos_diff / time_diff)


if __name__ == "__main__":
    main()
