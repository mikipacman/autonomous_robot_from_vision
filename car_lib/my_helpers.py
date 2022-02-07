import collections
import os
import time

import cv2
import numpy as np
import scipy.spatial.transform

# Settings for calibrating camera
chessboard_flags = None
# chessboard_flags = cv2.calib_cb_adaptive_thresh
# chessboard_flags += cv2.calib_cb_fast_check
# chessboard_flags += cv2.calib_cb_normalize_image
chessboard_shape = (8, 5)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# Aruco detector parameters
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
detectorParams = cv2.aruco.DetectorParameters_create()
detectorParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
MARKER_SIDE = 0.168


def get_calibration_parameters(path_to_images):
    paths = [os.path.join(path_to_images, i) for i in os.listdir(path_to_images)]

    # based on https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    cols, rows = chessboard_shape
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    for img_path in paths:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_shape, None)

        assert ret

        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    h, w = img.shape[:2]
    new_camera_mat, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (10 * w, 10 * h)
    )
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, new_camera_mat, (w, h), 5)
    return ret, mtx, dist, rvecs, tvecs, mapx, mapy, new_camera_mat, roi


def undistort_img(img, params):
    _, _, _, _, _, mapx, mapy, _, roi = params
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    x, y, w, h = roi
    dst = dst[y : y + h, x : x + w]
    return dst


def get_one_marker_vecs(img, params):
    _, _, dist, _, _, _, _, new_camera_mat, _ = params
    corners, ids = detect_aruco(img)

    if ids is None:
        return None

    assert ids is not None, "no markers detected!"
    assert len(ids) == 1, ids

    # draw id and boudaries
    cv2.aruco.drawDetectedMarkers(img, corners, ids)

    # This takes multiple corners and calculates 3D pose
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, MARKER_SIDE, new_camera_mat, dist
    )
    rvec, tvec = rvecs[0], tvecs[0]
    return rvec, tvec


def get_one_marker_vecs_until_success(cam, params):
    ret = None
    while ret is None:
        img = cam.get_frame()
        ret = get_one_marker_vecs(img, params)

    return ret


def detect_aruco(img):
    corners, ids, _ = cv2.aruco.detectMarkers(
        img, dictionary, None, None, detectorParams
    )
    return corners, ids


def rvec_to_angle(rvec):
    r = scipy.spatial.transform.Rotation.from_rotvec(rvec)
    return r.as_euler("xyz", degrees=True)[0][1]


def resize(img, scaling):
    dim = (int(img.shape[1] * scaling), int(img.shape[0] * scaling))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img


class FPS:
    def __init__(self, avarageof=50):
        self.frametimestamps = collections.deque(maxlen=avarageof)

    def __call__(self):
        self.frametimestamps.append(time.time())
        if len(self.frametimestamps) > 1:
            return len(self.frametimestamps) / (
                self.frametimestamps[-1] - self.frametimestamps[0]
            )
        else:
            return 0.0
