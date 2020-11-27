import cv2
import numpy as np
import os
from scipy.spatial.transform import Rotation

class ivr_vision:
    _blob_kernel_size = 5
    _target_template = cv2.imread(
        os.path.dirname(os.path.realpath(__file__)) + '/target_template.png',
        cv2.IMREAD_GRAYSCALE
    )
    _direction_correction = np.array([1.0, -1.0])  # Y-coordinates are flipped in cam feeds
    DEBUG = True
    YELLOW_RANGE = [(20, 100, 100), (30, 255, 255)]
    BLUE_RANGE = [(100, 150, 0), (140, 255, 255)]
    GREEN_RANGE = [(36, 25, 25), (70, 255, 255)]
    RED_RANGE = [(0, 70, 50), (10, 255, 255)]
    ORANGE_RANGE = [(10, 0, 0), (20, 255, 255)]  # NB: in HSV

    @staticmethod
    def _transform(a, d, alpha, theta, angle):
        x_rot = np.eye(4)
        x_trans = np.eye(4)
        z_trans = np.eye(4)
        z_rot = np.eye(4)
        x_rot[:-1, :-1] = Rotation.from_euler("xyz", [alpha, 0, 0]).as_matrix()
        x_trans[0, -1] = a
        z_rot[:-1, :-1] = Rotation.from_euler("xyz", [0, 0, theta + angle]).as_matrix()
        z_trans[-2, -1] = d
        return z_rot @ z_trans @ x_trans @ x_rot

    @staticmethod
    def debug_pose(joints):
        print('[' +
            f'Y({joints[0][0]:.2f}, {joints[0][1]:.2f}, {joints[0][2]:.2f}), ' +
            f'B({joints[1][0]:.2f}, {joints[1][1]:.2f}, {joints[1][2]:.2f}), ' +
            f'G({joints[2][0]:.2f}, {joints[2][1]:.2f}, {joints[2][2]:.2f}), ' +
            f'R({joints[3][0]:.2f}, {joints[3][1]:.2f}, {joints[3][2]:.2f})' +
        ']')

    @staticmethod
    def debug_angles(angles):
        print('[' +
            f'B_X({angles[0]:.2f}), ' +
            f'B_Y({angles[1]:.2f}), ' +
            f'G_X({angles[2]:.2f})' +
        ']')

    @staticmethod
    def fit_theta1(joints_3d, prev_theta1_estimate):
        best_angles = None
        best_error = float('inf')
        theta1_guesses = np.linspace(-np.pi, np.pi, 200)
        for theta1_guess in theta1_guesses:
            estimated_angles = ivr_vision._compute_joint_angles(joints_3d, theta1_guess)
            fk_joint_locs = ivr_vision._get_joint_locs_fk(estimated_angles)
            error = ivr_vision._theta1_estimate_error(truth=joints_3d, guess=fk_joint_locs)
            if error < best_error:
                best_error = error
                best_angles = estimated_angles
            # if ivr_vision.DEBUG:
            #     print(f'fitting locations with theta1={theta1_guess:.2f} gives error={error:.3f}')
        # find the best estimate close to previous one out of {theta1 - pi, theta1, theta1 + pi}
        _temp_angle = best_angles[0]
        options = [_temp_angle - np.pi, _temp_angle, _temp_angle + np.pi]
        options = [angle for angle in options if angle > -np.pi or angle <= np.pi]
        # check which is closest
        theta1_estimate = options[0]
        best_diff = float('inf')
        for option in options:
            diff = np.abs(option - prev_theta1_estimate)
            if diff < best_diff:
                best_diff = diff
                theta1_estimate = option
        # compute j2, j3, j4
        best_angles = ivr_vision._compute_joint_angles(joints_3d, theta1_estimate)
        fk_joint_locs = ivr_vision._get_joint_locs_fk(estimated_angles)
        return best_angles, ivr_vision._theta1_estimate_error(truth=joints_3d, guess=fk_joint_locs)

    @staticmethod
    def _theta1_estimate_error(truth, guess):
        return np.linalg.norm(truth - guess)

    # undo z-rotation and estimate angles
    @staticmethod
    def _compute_joint_angles(joint_locs, theta1_guess):
        # undo theta1
        _joint_locs = joint_locs.copy()
        for i, J in enumerate(_joint_locs):
            _joint_locs[i] = ivr_vision._rotate_around_z_axis(-theta1_guess, J)
        _angles = ivr_vision.compute_joint_angles(_joint_locs)
        # if ivr_vision.DEBUG:
        #     ivr_vision.debug_angles(_angles)
        return np.insert(_angles, 0, [theta1_guess])

    def _get_joint_locs_fk(angles):
        # simplified forward kinematics for task 4.1
        _mat_1 = ivr_vision._transform(theta=np.pi/2, a=0.0, d=2.5, alpha=np.pi/2 , angle=angles[0])
        _mat_2 = ivr_vision._transform(theta=np.pi/2, a=0.0, d=0.0, alpha=np.pi/2 , angle=angles[1])
        _mat_3 = ivr_vision._transform(theta=0.0    , a=3.5, d=0.0, alpha=-np.pi/2, angle=angles[2])
        _mat_4 = ivr_vision._transform(theta=0.0    , a=3.0, d=0.0, alpha=0.0     , angle=angles[3])
        # compute locations given angles
        J0 = np.array([0.0, 0.0, 0.0])
        J1 = (_mat_1)[:-1, -1]
        J2 = (_mat_1 @ _mat_2 @ _mat_3)[:-1, -1]
        J3 = (_mat_1 @ _mat_2 @ _mat_3 @ _mat_4)[:-1, -1]
        # if ivr_vision.DEBUG:
        #     ivr_vision.debug_pose(np.array([J0, J1, J2, J3]))
        return np.array([J0, J1, J2, J3])

    @staticmethod
    def compute_joint_angles(joint_locs):
        xy_norm = np.array([0.0, 0.0, 1.0])
        xz_norm = np.array([0.0, 1.0, 0.0])
        yz_norm = np.array([1.0, 0.0, 0.0])

        # links 2, 3, 4 respectively
        joint_angles = np.array([0.0, 0.0, 0.0])
        # link 2: blue, around X-axis
        B2G = joint_locs[2] - joint_locs[1]
        joint_angles[0] = np.arctan2(B2G[2], B2G[1]) - np.pi / 2.0
        # link 3: blue, around Y-axis
        B2G = ivr_vision._rotate_around_x_axis(-joint_angles[0], B2G)  # make relative
        joint_angles[1] = np.arctan2(B2G[0], B2G[2])
        # link 4: green, around X-axis
        # B2G = ivr_vision._rotate_around_x_axis(joint_angles[0], B2G)  # make relative
        G2R = joint_locs[3] - joint_locs[2]
        joint_angles[2] = np.arctan2(G2R[2], G2R[1]) - joint_angles[0] - np.pi / 2.0

        if ivr_vision.DEBUG:
            #ivr_vision.debug_pose(joint_locs)
            # ivr_vision.debug_angles(joint_angles)
            pass
        return joint_angles

    @staticmethod
    def _combine_2d_to_3d(yz_2d, xz_2d):
        return np.array([
            xz_2d[0],
            yz_2d[0],
            (yz_2d[1] + xz_2d[1]) / 2.0
        ])

    @staticmethod
    def combine_joint_locations(cam1_locations_2d, cam2_locations_2d):
        """cam1 is looking at YZ, cam2 is looking at XZ"""
        coords = np.repeat(-1.0, 4 * 3).reshape(4, -1)
        for i in range(cam1_locations_2d.shape[0]):
            coords[i] = ivr_vision._combine_2d_to_3d(cam1_locations_2d[i], cam2_locations_2d[i])
        return coords

    @staticmethod
    def update_joint_locations(image, locations):
        """detects joint locations in meters in orthogonal plane"""
        yellow = ivr_vision.detect_blob(image, ivr_vision.YELLOW_RANGE)
        blue   = ivr_vision.detect_blob(image, ivr_vision.BLUE_RANGE)
        green  = ivr_vision.detect_blob(image, ivr_vision.GREEN_RANGE)
        red    = ivr_vision.detect_blob(image, ivr_vision.RED_RANGE)
        p2m    = ivr_vision._pixel2meter(yellow, blue, 2.5)
        center = p2m * yellow
        if ivr_vision.DEBUG:
            # draw dots in the centers to check that this works
            r = 5
            try:
                cv2.circle(image, tuple(yellow), r, ivr_vision.invert(ivr_vision.YELLOW_RANGE[1]), -1)
                cv2.circle(image, tuple(blue), r, ivr_vision.invert(ivr_vision.BLUE_RANGE[1]), -1)
                cv2.circle(image, tuple(green), r, ivr_vision.invert(ivr_vision.GREEN_RANGE[1]), -1)
                cv2.circle(image, tuple(red), r, ivr_vision.invert(ivr_vision.RED_RANGE[1]), -1)
            except Exception as e:
                print("could not display debug circles")
        # update any new locations we found
        if yellow is not None:
            locations[0] = np.multiply(p2m * yellow - center, ivr_vision._direction_correction)
        if blue is not None:
            locations[1] = np.multiply(p2m * blue - center, ivr_vision._direction_correction)
        if green is not None:
            locations[2] = np.multiply(p2m * green - center, ivr_vision._direction_correction)
        if red is not None:
            locations[3] = np.multiply(p2m * red - center, ivr_vision._direction_correction)

    @staticmethod
    def _pixel2meter(joint1, joint2, real_dist):
        pixel_dist = np.linalg.norm(joint1 - joint2)
        return real_dist / pixel_dist

    @staticmethod
    def detect_blob(image, hsv_range):
        kernel = np.ones((ivr_vision._blob_kernel_size, ivr_vision._blob_kernel_size), np.uint8)
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        inRange = cv2.inRange(img_hsv, hsv_range[0], hsv_range[1])
        mask = cv2.dilate(
            inRange,
            kernel,
            iterations=3
        )
        M = cv2.moments(mask)
        if M['m00'] == 0.0:
            return None  # occlusion
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return np.array([cx, cy])

    @staticmethod
    def detect_target(image):
        # detection of target
        template_size = 26.0
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        thresholded = cv2.inRange(img_hsv, ivr_vision.ORANGE_RANGE[0], ivr_vision.ORANGE_RANGE[1])
        total = np.sum(np.sum(thresholded))
        if total == 0.0:
            return None  # target is occluded by something else
        match = cv2.matchTemplate(thresholded, ivr_vision._target_template, 1)
        best_val, _, best_position, _ = cv2.minMaxLoc(match)
        if ivr_vision.DEBUG:
            # im_debug=cv2.imshow('debug', match)
            # print(best_val)
            pass
        if best_val > 0.6:
            return None  # target is occluded by something else
        # find center
        cx = best_position[0] + template_size / 2.0
        cy = best_position[1] + template_size / 2.0
        target = np.array([int(cx), int(cy)])
        if ivr_vision.DEBUG:
            cv2.circle(image, tuple(target), 5, ivr_vision.invert(ivr_vision.ORANGE_RANGE[1]), -1)
        # scaling
        yellow = ivr_vision.detect_blob(image, ivr_vision.YELLOW_RANGE)
        blue   = ivr_vision.detect_blob(image, ivr_vision.BLUE_RANGE)
        p2m = ivr_vision._pixel2meter(yellow, blue, 2.5)
        return np.multiply(p2m * (target - yellow), ivr_vision._direction_correction)

    @staticmethod
    def invert(color):
        return (255 - color[0], 255 - color[1], 255 - color[2])

    @staticmethod
    def _vector_angle(v1, v2):
        return np.arccos(np.dot(v1, v2) / (np.dot(v1, v1) * np.dot(v2, v2)))

    @staticmethod
    def _project_vector(v, n):
        return v - np.multiply(np.dot(v, n) / np.dot(v, v), v)

    @staticmethod
    def _rotate_around_x_axis(angle, v):
        M = np.array([
            [1.0, 0.0          ,  0.0          ],
            [0.0, np.cos(angle), -np.cos(angle)],
            [0.0, np.sin(angle),  np.cos(angle)]
        ])
        return np.dot(M, v)

    @staticmethod
    def _rotate_around_z_axis(angle, v):
        M = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle),  np.cos(angle), 0.0],
            [0.0          ,  0.0          , 1.0]
        ])
        return np.dot(M, v)


class camera:
    def __init__(self, position):
        self._p = position
