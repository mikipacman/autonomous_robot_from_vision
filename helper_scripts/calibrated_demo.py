import cv2

from car_lib.cars import RESOLUTIONS, Camera, Connection
from car_lib.my_helpers import (chessboard_flags, chessboard_shape, criteria,
                                get_calibration_parameters, undistort_img)


def main():
    cv2.namedWindow("original")
    cv2.namedWindow("undistorted")
    connection = Connection()
    cam = Camera(connection=connection)
    w, h = RESOLUTIONS[cam.get_quality()]
    assert w == 640 and h == 480
    folder_path = "images_for_calibration/640x480"

    # init undist
    params = get_calibration_parameters(folder_path)

    while True:
        connection.keep_stream_alive()
        img = cam.get_frame()
        undist = undistort_img(img, params)
        cv2.imshow("original", img)
        cv2.imshow("undistorted", undist)
        cv2.pollKey()


if __name__ == "__main__":
    main()
