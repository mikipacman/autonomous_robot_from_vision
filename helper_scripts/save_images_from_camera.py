#!/usr/bin/env python3
import os
import time

import cv2
from car_lib.cars import RESOLUTIONS, Camera, Connection
from car_lib.my_helpers import chessboard_flags, chessboard_shape, criteria


def main():
    cv2.namedWindow("demo")
    connection = Connection()
    cam = Camera(connection=connection)

    num_images_to_save = 20
    images_saved = 0
    images = []
    quality = RESOLUTIONS[cam.get_quality()]
    folder_path = f"a/images_for_calibration/{quality[0]}x{quality[1]}"

    os.makedirs(folder_path, exist_ok=True)

    while images_saved < num_images_to_save:
        connection.keep_stream_alive()

        # get Img
        img = cam.get_frame()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Try chessboard
        ret, corners = cv2.findChessboardCorners(
            gray, chessboard_shape, flags=chessboard_flags
        )

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_to_save = img.copy()
            cv2.drawChessboardCorners(img, (8, 5), corners2, ret)
            cv2.imshow("demo", img)

            if ord("s") == cv2.waitKey() & 0xFF:
                path = os.path.join(folder_path, f"{images_saved}.jpg")
                cv2.imwrite(path, img_to_save)
                images_saved += 1
                print("images saved", images_saved)
        else:
            cv2.imshow("demo", img)
            cv2.pollKey()


if __name__ == "__main__":
    main()
