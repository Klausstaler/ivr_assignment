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


class camera:
    def __init__(self, position):
        self._p = position
