import cv2
import numpy as np

from car_lib.cars import RESOLUTIONS, Camera, Connection
from car_lib.my_helpers import (chessboard_flags, chessboard_shape, criteria,
                                get_calibration_parameters, undistort_img)


def coordinates(point):
    return [int(i) for i in tuple(point.ravel())]


def draw(img, corners, imgpts):
    corner = coordinates(corners[0].ravel())
    img = cv2.line(img, corner, coordinates(imgpts[0]), (255, 0, 0), 5)
    img = cv2.line(img, corner, coordinates(imgpts[1]), (0, 255, 0), 5)
    img = cv2.line(img, corner, coordinates(imgpts[2]), (0, 0, 255), 5)
    return img


def main():
    cv2.namedWindow("original")
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
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, chessboard_shape, None)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(img, (8, 5), corners2, ret)
            ret, mtx, dist, rvecs, tvecs, mapx, mapy, new_camera_mat, roi = params

            objp = np.zeros((8 * 5, 3), np.float32)
            objp[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2)
            _, rvec, tvec = cv2.solvePnP(
                objectPoints=objp,
                imagePoints=corners2,
                cameraMatrix=new_camera_mat,
                distCoeffs=dist,
            )

            imgpts, _ = cv2.projectPoints(
                objectPoints=np.array(
                    [
                        [1, 0, 0],
                        [0, 2, 0],
                        [0, 0, -3],
                    ],
                    dtype=np.float64,
                ),
                rvec=rvec,
                tvec=tvec,
                cameraMatrix=new_camera_mat,
                distCoeffs=dist,
            )
            print(imgpts)
            draw(img, corners2, imgpts)

        cv2.imshow("original", img)
        cv2.pollKey()


if __name__ == "__main__":
    main()
