import cv2
import numpy as np

class ivr_vision:
    _blob_kernel_size = 5
    DEBUG = True
    YELLOW_RANGE = [(0, 100, 100), (0, 255, 255)]
    BLUE_RANGE = [(100, 0, 0), (255, 0, 0)]
    GREEN_RANGE = [(0, 100, 0), (0, 255, 0)]
    RED_RANGE = [(0, 0, 100), (0, 0, 255)]
    XY_NORM = np.array([0.0, 0.0, 1.0])
    XZ_NORM = np.array([0.0, 1.0, 0.0])
    YZ_NORM = np.array([1.0, 0.0, 0.0])

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
    def _intersect_projection_planes(dir1, axis1, dir2, axis2):
        norm1 = np.cross(dir1, axis1)
        norm2 = np.cross(dir2, axis2)
        return np.cross(norm1, norm2)  # this is the direction of the intersecting line


    @staticmethod
    def compute_end_effector_location(angles):
        b_pos = np.array([0.0, 0.0, 2.5])  # by definition

        # compute green joint location
        B2G_len = 3.5
        B2G_yz = np.array([0.0, B2G_len * np.sin(angles[0]), B2G_len * np.cos(angles[0])])
        B2G_xz = np.array([B2G_len * np.sin(angles[1]), 0.0, B2G_len * np.cos(angles[1])])
        B2G_dir = ivr_vision._intersect_projection_planes(
            B2G_yz, ivr_vision.YZ_NORM,
            B2G_xz, ivr_vision.XZ_NORM
        )
        # length and direction of planes intersection allows us to determine 3D location
        t = np.sqrt(np.square(B2G_len) / np.dot(B2G_dir, B2G_dir))
        B2G = t * B2G_dir
        g_pos = b_pos + B2G

        if ivr_vision.DEBUG:
            print(f'EE({g_pos[0]:.1f}, {g_pos[1]:.1f}, {g_pos[2]:.1f})')

        return g_pos  # todo: red location

    @staticmethod
    def compute_joint_angles(joint_locs):
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
            # ivr_vision.debug_pose(joint_locs)
            # ivr_vision.debug_angles(joint_angles)
            pass
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
        direction_correction = np.array([1.0, -1.0])  # Y-coordinates are flipped in cam feeds
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
            np.multiply(p2m * yellow - center, direction_correction),
            np.multiply(p2m * blue - center, direction_correction),
            np.multiply(p2m * green - center, direction_correction),
            np.multiply(p2m * red - center, direction_correction)
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

    @staticmethod
    def _project_vector(v, n):
        return v - np.multiply(np.dot(v, n) / np.dot(v, v), v)

    @staticmethod
    def _rotate_around_x_axis(angle, v):
        M = np.array([
            [1.0, 0.0          , 0.0                 ],
            [0.0, np.cos(angle), -1.0 * np.cos(angle)],
            [0.0, np.sin(angle),        np.cos(angle)]
        ])
        return np.dot(M, v)


class camera:
    def __init__(self, position):
        self._p = position
