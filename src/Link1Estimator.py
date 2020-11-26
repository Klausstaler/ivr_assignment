#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float64, Float64MultiArray
from sensor_msgs.msg import JointState
import matplotlib.pyplot as plt
from forward_kinematics import robot

class Link1Estimator:

    def __init__(self, robot):
        self.robot = robot
        self.prev_angle = 0.0
        self.desired_position = np.array([0.0, 0.0, 9.0])

    def links_cb(self, data, desired_position):
        self.desired_position = desired_position
        if type(data) == JointState:
            angles = data.position
            self.robot.link2.angle = angles[1]
            self.robot.link3.angle = angles[2]
            self.robot.link4.angle = angles[3]
            self.robot.link1.angle = self.estimate_link1()
        else:
            self.robot.link2.angle = data.data[0]
            self.robot.link3.angle = data.data[1]
            self.robot.link4.angle = data.data[2]
            self.robot.link1.angle = self.estimate_link1()
        return np.array([self.robot.link1.angle, self.robot.link2.angle, self.robot.link3.angle, self.robot.link4.angle])

    def estimate_link1(self):

        best_angle, min_err = 0.0, 20
        all_errors = []
        all_errors2 = []
        pos_d = self.desired_position
        for curr_angle in np.linspace(-np.pi, np.pi, num=180):
            self.robot.link1.angle = curr_angle
            pos = self.robot.update_effector_estimate()  # actual position
            error = np.linalg.norm(pos - pos_d)
            #curr_angle = self.normalize_angle(curr_angle)
            all_errors.append((curr_angle, error))
            all_errors2.append((error, curr_angle))
        all_errors2.sort()
        best_angle = sum([angle_err[1] + np.pi for angle_err in all_errors2[:3]]) # take average
        best_angle = best_angle / 3 - np.pi

        all_errors.sort()
        #plt.scatter([x[0] for x in all_errors], [x[1] for x in all_errors])
        #plt.show()
        best_angle = self.normalize_angle(best_angle)
        return best_angle

    def normalize_angle(self, curr_angle, offset=0.0):
        curr_angle -= offset
        multiple = curr_angle // (2 * np.pi)
        curr_angle = curr_angle - 2 * np.pi * multiple
        curr_angle += offset
        while curr_angle > np.pi or curr_angle < -np.pi:
            curr_angle -= 2 * np.pi if curr_angle > np.pi else -2 * np.pi
        return curr_angle
