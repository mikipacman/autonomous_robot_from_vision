import time

import cv2
import numpy as np

from car_lib.cars import RESOLUTIONS, Camera, Connection, Direction, Motors
from car_lib.my_helpers import (FPS, MARKER_SIDE, detect_aruco,
                                get_calibration_parameters,
                                get_one_marker_vecs,
                                get_one_marker_vecs_until_success, resize,
                                rvec_to_angle)


class Robot:
    def __init__(
        self,
        flash=True,
    ):
        connection = Connection()
        cam = Camera(connection=connection)
        motors = Motors(connection=connection, motor2_multiplier=0.99)

        self.window_name = "robot"
        self.cam = cam
        self.motors = motors
        self.connection = connection
        self.flash = flash
        if self.flash:
            self.cam.flash_on()

        cv2.namedWindow(self.window_name)
        w, h = RESOLUTIONS[cam.get_quality()]
        assert w == 640 and h == 480
        folder_path = "images_for_calibration/640x480"
        params = get_calibration_parameters(folder_path)
        ret, mtx, dist, rvecs, tvecs, mapx, mapy, new_camera_mat, roi = params
        self.camera_params = params
        self.height = h
        self.width = w

        # Robot params
        ## Powers
        self.rotation_power = 62
        self.drive_power = 66
        ## Sleeps during discrete movements
        self.sleep_time_during_rotation = 0.45
        self.sleep_time_during_driving = 0.3
        ## Speed of approximate continous movements
        self.drive_velocity_unit_per_sec = 0.23
        self.rotate_velocity_unit_per_sec = 66
        ## Other
        self.marker_discovery_range = (0.35, 0.65)
        ## Servo
        self.servo_hidden_position = 2120
        self.servo_pickup_position = 1500
        self.servo_straight_position = 1000
        self.servo_current_position = self.servo_hidden_position
        self.motors.command_servo(self.servo_current_position)

        min_range, max_range = self.marker_discovery_range
        px_min_range = min_range * self.width
        px_max_range = max_range * self.width
        self.px_max_range = int(px_max_range)
        self.px_min_range = int(px_min_range)
        self.fps = FPS()

        self.debug = True

    def find_any_marker(self, marker_ids, start_dir):
        # Rotate until you find any marker
        assert start_dir is Direction.LEFT or start_dir is Direction.RIGHT
        ret = self._find_marker_decision(marker_ids)
        while True:
            if ret is None:
                self.rotate(start_dir)
            else:
                _, ids = ret
                if any([m in ids for m in marker_ids]):
                    return list([x[0] for x in ids])
                else:
                    self.rotate(start_dir)
            ret = self._find_marker_decision(marker_ids)

    def find_marker(self, marker_id, start_dir):
        # Rotate until you find a specific marker
        assert start_dir is Direction.LEFT or start_dir is Direction.RIGHT
        ret = self._find_marker_decision(marker_id)
        while True:
            if ret is None:
                self.rotate(start_dir)
            else:
                corners, ids = ret
                assert marker_id in ids
                marker_idx = list(ids).index(marker_id)
                marker_x_center = corners[marker_idx].mean(axis=1)[0][0]

                if self.px_min_range < marker_x_center < self.px_max_range:
                    rvec, tvec = self._get_position_of_marker(corners[marker_idx])
                    angle = rvec_to_angle(rvec)
                    dist = np.linalg.norm(tvec)
                    return angle, dist
                elif self.px_max_range < marker_x_center:
                    self.rotate(Direction.RIGHT)
                else:
                    self.rotate(Direction.LEFT)

            ret = self._find_marker_decision(marker_id)

    def rotate(self, dir):
        self.motors.command(self.rotation_power, dir)
        self.sleep(self.sleep_time_during_rotation)

    def rotate_approximately(self, angle):
        dir = Direction.LEFT if angle < 0 else Direction.RIGHT
        time_to_drive = np.abs(angle) / self.rotate_velocity_unit_per_sec
        start_time = time.time()
        while time.time() - start_time < time_to_drive:
            self.motors.command(self.drive_power, dir)
            self._get_image()

    def drive_straight(self, dir=Direction.FORWARD):
        self.motors.command(self.drive_power, dir)
        self.sleep(self.sleep_time_during_driving)

    def drive_straight_appoximately(self, distance, dir=Direction.FORWARD):
        time_to_drive = distance / self.drive_velocity_unit_per_sec
        start_time = time.time()
        while time.time() - start_time < time_to_drive:
            self.motors.command(self.drive_power, dir)
            self._get_image()

    def sleep(self, time_in_sec):
        start = time.time()
        while time.time() - start < time_in_sec:
            self._get_image()

    def blink(self):
        for _ in range(3):
            self.cam.flash_on()
            self.sleep(0.05)
            self.cam.flash_off()
            self.sleep(0.05)

        if self.flash:
            self.cam.flash_on()

    def servo_hide(self):
        self._servo_move_to_pos(self.servo_hidden_position)

    def servo_pickup(self):
        self._servo_move_to_pos(self.servo_pickup_position)

    def servo_straight(self):
        self._servo_move_to_pos(self.servo_straight_position)

    def get_baskets_params(self):
        img = self._get_image()
        baskets_params = self._get_baskets_params(img)
        for params in baskets_params:
            params["left"] /= self.width
            params["right"] /= self.width
            params["top"] /= self.height
            params["bottom"] /= self.height
        return baskets_params

    def drive_to_pos_in_marker_coord(self, marker_id, pos):
        ret = None
        while ret is None:
            ret = self._find_marker_decision(marker_id)
        corners, ids = ret
        assert marker_id in ids
        marker_idx = list(ids).index(marker_id)
        rvec, tvec = self._get_position_of_marker(corners[marker_idx])

        point_in_marker_coord = np.array(pos)
        rot_mat = cv2.Rodrigues(rvec)[0]
        point_in_camera_coord = rot_mat @ point_in_marker_coord + tvec

        x, y, z = list(point_in_camera_coord[0])
        angle = np.sign(x) * np.arctan(np.abs(x / z)) / 3.14 * 180
        dist = np.linalg.norm(point_in_camera_coord)

        return angle, dist

    def _get_baskets_params(self, img):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        red_mask = self._get_red_mask(hsv_img)
        green_mask = self._get_green_mask(hsv_img)

        green_score = green_mask.sum()
        red_score = red_mask.sum()

        baskets = []
        threshold = 5e5

        if green_score > threshold:
            params = self._detect_basket_mask_lines(green_mask)
            params["color"] = "green"
            params["score"] = green_score
            params["mask"] = green_mask
            baskets.append(params)

        if red_score > threshold:
            params = self._detect_basket_mask_lines(red_mask)
            params["color"] = "red"
            params["score"] = red_score
            params["mask"] = red_mask
            baskets.append(params)

        return baskets

    def _detect_basket_mask_lines(self, mask):
        h_threshold = 50 * 255
        v_threshold = 60 * 255
        left = np.argmax(mask.sum(axis=0) > v_threshold)
        right = mask.shape[1] - np.argmax(mask.sum(axis=0)[::-1] > v_threshold)
        top = np.argmax(mask.sum(axis=1) > h_threshold)
        bottom = mask.shape[0] - np.argmax(mask.sum(axis=1)[::-1] > h_threshold)
        return {"left": left, "right": right, "top": top, "bottom": bottom}

    def _get_red_mask(self, hsv_img):
        # positive red hue margin
        lower1 = np.array([0, 110, 150])
        upper1 = np.array([15, 255, 255])
        mask1 = cv2.inRange(hsv_img, lower1, upper1)

        # negative red hue margin
        lower2 = np.array([155, 110, 150])
        upper2 = np.array([179, 255, 255])
        mask2 = cv2.inRange(hsv_img, lower2, upper2)

        red_mask = mask1 + mask2
        return red_mask

    def _get_green_mask(self, hsv_img):
        # green hue margin
        lower = np.array([45, 90, 130])
        upper = np.array([75, 255, 255])
        green_mask = cv2.inRange(hsv_img, lower, upper)
        return green_mask

    def _servo_move_to_pos(self, pos):
        curr = self.servo_current_position
        positions = list(range(min(pos, curr), max(pos, curr), 10))
        if curr > pos:
            positions = reversed(positions)

        for pos in positions:
            self.motors.command_servo(pos)
            self._get_image()

        self.servo_current_position = pos

    def _flip_dir(self, dir):
        assert dir is Direction.LEFT or dir is Direction.RIGHT
        if dir is Direction.LEFT:
            return Direction.RIGHT
        else:
            return Direction.LEFT

    def _get_position_of_marker(self, marker_corners):
        _, _, dist, _, _, _, _, new_camera_mat, _ = self.camera_params
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            [marker_corners], MARKER_SIDE, new_camera_mat, dist
        )
        rvec, tvec = rvecs[0], tvecs[0]
        return rvec, tvec

    def _get_angle_to_marker(self, marker_corners):
        rvec, _ = self._get_position_of_marker(marker_corners)
        return rvec_to_angle(rvec)

    def _find_marker_decision(self, marker_id):
        # Return None if we should rotate or corners and ids if marker is found
        img = self._get_image()
        ret = self._get_marker_from_image(img)
        if ret is None:
            return None
        else:
            corners, ids = ret
            if marker_id not in ids:
                return None
            else:
                return corners, ids

    def _get_image(self):
        self.connection.keep_stream_alive()
        img = self.cam.get_frame()

        if self.debug:
            # Draw range lines
            def draw_line(start, end, img, col):
                return cv2.line(img, start, end, col, 1)

            def draw_v_line(x, img, col):
                start = (x, 0)
                end = (x, self.height)
                return draw_line(start, end, img, col)

            def draw_h_line(x, img, col):
                start = (0, x)
                end = (self.width, x)
                return draw_line(start, end, img, col)

            def draw(img, text, pos):
                return cv2.putText(
                    img,
                    text,
                    pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            def draw_mask(img, mask, col):
                background = np.full(img.shape, col, dtype=np.uint8)
                bk = cv2.bitwise_or(background, background, mask=mask)
                mask = cv2.bitwise_not(mask)
                fg = cv2.bitwise_or(img, img, mask=mask)
                return cv2.bitwise_or(fg, bk)

            RED = (0, 0, 255)
            GREEN = (0, 255, 0)
            BLUE = (255, 0, 0)

            img_to_draw = img.copy()
            img_to_draw = draw_v_line(self.px_min_range, img_to_draw, BLUE)
            img_to_draw = draw_v_line(self.px_max_range, img_to_draw, BLUE)
            img_to_draw = draw(img_to_draw, f"fps={int(self.fps())}", (5, 30))

            basket_params = self._get_baskets_params(img)

            for params in basket_params:
                assert params["color"] in ("red", "green")
                color = RED if params["color"] == "red" else GREEN
                img_to_draw = draw_v_line(params["left"], img_to_draw, color)
                img_to_draw = draw_v_line(params["right"], img_to_draw, color)
                img_to_draw = draw_h_line(params["top"], img_to_draw, color)
                img_to_draw = draw_h_line(params["bottom"], img_to_draw, color)
                img_to_draw = draw_mask(img_to_draw, params["mask"], color)

            # Draw markers
            ret = self._get_marker_from_image(img)
            if ret:
                corners, ids = ret
                cv2.aruco.drawDetectedMarkers(img_to_draw, corners, ids)

                # draw axis
                _, _, dist, _, _, _, _, new_camera_mat, _ = self.camera_params
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, MARKER_SIDE, new_camera_mat, dist
                )

                for i, (rvec, tvec, id) in enumerate(zip(rvecs, tvecs, ids)):
                    cv2.aruco.drawAxis(
                        img_to_draw, new_camera_mat, dist, rvec, tvec, 0.3
                    )
                    # draw text
                    angle = rvec_to_angle(rvec)
                    dist = np.linalg.norm(tvec)
                    move = i * 3 * 40 + 90
                    img_to_draw = draw(img_to_draw, f"angle={int(angle)}", (5, move))
                    img_to_draw = draw(
                        img_to_draw, f"dist={round(dist, 2)}", (5, 30 + move)
                    )
                    img_to_draw = draw(img_to_draw, f"id={id[0]}", (5, 60 + move))

        else:
            img_to_draw = img
        cv2.imshow(self.window_name, resize(img_to_draw, 1.2))
        key = cv2.pollKey()

        if self.debug:
            if key == ord("p"):
                cv2.waitKey(0)
            elif key == ord("q"):
                exit(0)
        return img

    def _get_marker_from_image(self, img):
        corners, ids = detect_aruco(img)
        if ids is None:
            return None
        else:
            assert len(ids) > 0
            return corners, ids

    def __del__(self):
        self.cam.flash_off()
        self.servo_hide()
