import cv2
import numpy as np
import scipy.spatial.transform
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

    # Aruco detector parameters
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    detectorParams = cv2.aruco.DetectorParameters_create()
    detectorParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR

    # Note that unit is not specified, we just need to stick to one (here meters)
    MARKER_SIDE = 0.168
    ret, mtx, dist, rvecs, tvecs, mapx, mapy, new_camera_mat, roi = params

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    DARK_GREEN = (0, 127, 0)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    VIOLET = (255, 0, 255)
    CYAN = (255, 255, 0)
    YELLOW = (0, 255, 255)
    ORANGE = (0, 100, 255)

    def calc_obj_pts(cube_side=MARKER_SIDE):
        ret = np.array(
            [
                [-0.5, -0.5, 0],
                [0.5, -0.5, 0],
                [0.5, 0.5, 0],
                [-0.5, 0.5, 0],
                [-0.5, -0.5, 1],
                [0.5, -0.5, 1],
                [0.5, 0.5, 1],
                [-0.5, 0.5, 1],
            ]
        )
        return cube_side * ret

    while True:
        connection.keep_stream_alive()
        img = cam.get_frame()

        # detect
        corners, ids, _ = cv2.aruco.detectMarkers(
            img, dictionary, None, None, detectorParams
        )

        if ids is not None:
            # draw id and boudaries
            cv2.aruco.drawDetectedMarkers(img, corners, ids)

            # This takes multiple corners and calculates 3D pose
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, MARKER_SIDE, new_camera_mat, dist
            )

            # draw axis
            for rvec, tvec in zip(rvecs, tvecs):
                cv2.aruco.drawAxis(img, new_camera_mat, dist, rvec, tvec, 0.1)

                # # draw cubes
                # objpts = calc_obj_pts()

                # for rvec, tvec in zip(rvecs, tvecs):
                #     imgpts = np.rint(
                #         cv2.projectPoints(
                #             objpts,
                #             rvec,
                #             tvec,
                #             new_camera_mat,
                #             0,
                #         )[0]
                #     ).astype(int)
                #     imgpts = imgpts.reshape((-1, 2))
                #     back = np.array([imgpts[:4]])
                #     front = np.array([imgpts[4:]])
                #     sides = []
                #     for i in range(4):
                #         sides.append([imgpts[i], imgpts[i + 4]])
                #     sides = np.array(sides)

                #     cv2.drawContours(img, back, -1, BLUE, cv2.FILLED)
                #     cv2.drawContours(img, sides, -1, GREEN, 3)
                #     cv2.drawContours(img, front, -1, RED, 3)

                r = scipy.spatial.transform.Rotation.from_rotvec(rvec)
                print(r.as_euler("xyz", degrees=True)[0][1])

        cv2.imshow("original", img)
        cv2.pollKey()


if __name__ == "__main__":
    main()
