import cv2
import numpy as np

class ivr_vision:
    _blob_kernel_size = 5
    DEBUG = True
    YELLOW_RANGE = [(0, 100, 100), (0, 255, 255)]
    BLUE_RANGE = [(100, 0, 0), (255, 0, 0)]
    GREEN_RANGE = [(0, 100, 0), (0, 255, 0)]
    RED_RANGE = [(0, 0, 100), (0, 0, 255)]

    @staticmethod
    def debug_pose(joints):
        print('[' +
            f'Y({joints[0][0]:.2f}, {joints[0][1]:.2f}, {joints[0][2]:.2f}), ' +
            f'B({joints[1][0]:.2f}, {joints[1][1]:.2f}, {joints[1][2]:.2f}), ' +
            f'G({joints[2][0]:.2f}, {joints[2][1]:.2f}, {joints[2][2]:.2f}), ' +
            f'R({joints[3][0]:.2f}, {joints[3][1]:.2f}, {joints[3][2]:.2f})' +
        ']')

    @staticmethod
    def compute_joint_angles(joint_locs):
        if ivr_vision.DEBUG:
            ivr_vision.debug_pose(joint_locs)
        # add dummy joint
        base = joint_locs[0]
        joint_locs = np.insert(joint_locs, 0, np.array([base[0], base[1], base[2] - 100.0]), axis=0)
        joint_angles = np.repeat(-1.0, 3)
        # blue joint angle
        blue_green = -1.0 * (joint_locs[3] - joint_locs[2])
        blue_yellow = -1.0 * (joint_locs[1] - joint_locs[2])
        angle = ivr_vision._vector_angle(blue_green, blue_yellow)
        # print(f'blue angle: {angle:.2f}, BG: {blue_green}, BY: {blue_yellow}')
        return joint_angles

    @staticmethod
    def combine_joint_locations(cam1_locations_2d, cam2_locations_2d):
        """cam1 is looking at YZ, cam2 is looking at XZ"""
        coords = np.repeat(-1.0, 4 * 3).reshape(4, -1)
        for i in range(cam1_locations_2d.shape[0]):
            coords[i, 0] = cam2_locations_2d[i, 0]
            coords[i, 1] = cam1_locations_2d[i, 0]
            coords[i, 2] = (cam1_locations_2d[i, 1] + cam2_locations_2d[i, 1]) / 2.0
        return coords

    @staticmethod
    def detect_joint_locations(image):
        """detects joint locations in meters in orthogonal plane"""
        yellow = ivr_vision.detect_blob(image, ivr_vision.YELLOW_RANGE)
        blue   = ivr_vision.detect_blob(image, ivr_vision.BLUE_RANGE)
        green  = ivr_vision.detect_blob(image, ivr_vision.GREEN_RANGE)
        red    = ivr_vision.detect_blob(image, ivr_vision.RED_RANGE)
        p2m    = ivr_vision._pixel2meter(yellow, blue, 2.5)
        center = p2m * yellow
        if ivr_vision.DEBUG:
            r = 5
            # draw dots in the centers to check that this works
            cv2.circle(image, tuple(yellow), r, ivr_vision.invert(ivr_vision.YELLOW_RANGE[1]), -1)
            cv2.circle(image, tuple(blue), r, ivr_vision.invert(ivr_vision.BLUE_RANGE[1]), -1)
            cv2.circle(image, tuple(green), r, ivr_vision.invert(ivr_vision.GREEN_RANGE[1]), -1)
            cv2.circle(image, tuple(red), r, ivr_vision.invert(ivr_vision.RED_RANGE[1]), -1)
        return np.array([
            -1.0 * (p2m * yellow - center),
            -1.0 * (p2m * blue - center),
            -1.0 * (p2m * green - center),
            -1.0 * (p2m * red - center)
        ])

    @staticmethod
    def _pixel2meter(joint1, joint2, real_dist):
        pixel_dist = np.linalg.norm(joint1 - joint2)
        return real_dist / pixel_dist

    @staticmethod
    def detect_blob(image, rgb_range):
        kernel = np.ones((ivr_vision._blob_kernel_size, ivr_vision._blob_kernel_size), np.uint8)
        mask = cv2.dilate(
            cv2.inRange(image, rgb_range[0], rgb_range[1]),
            kernel,
            iterations=3
        )
        M = cv2.moments(mask)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return np.array([cx, cy])

    @staticmethod
    def invert(color):
        return (255 - color[0], 255 - color[1], 255 - color[2])

    @staticmethod
    def _vector_angle(v1, v2):
        return np.arccos(np.dot(v1, v2) / (np.dot(v1, v1) * np.dot(v2, v2)))


class camera:
    def __init__(self, position):
        self._p = position
