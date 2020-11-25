#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import JointState
import matplotlib.pyplot as plt
from forward_kinematics import Link, robot


class Link1Estimator:

    def __init__(self):
        rospy.init_node('Link1Estimator', anonymous=True)
        # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data

        self.link_sub = rospy.Subscriber("/robot/joint_states", JointState, self.links_cb)
        # self.link_sub = rospy.Subscriber("/robot/joint_angles", Float64MultiArray, self.links_cb)

        self.actual_link1 = 0.0
        self.prev_angle = 0.0

    def links_cb(self, data):
        if type(data) == JointState:
            angles = data.position
            robot.link2.angle = angles[1]
            robot.link3.angle = angles[2]
            robot.link4.angle = angles[3]
            self.actual_link1 = angles[0]
            if np.linalg.norm(self.actual_link1 - self.prev_angle) > 0.01:
                robot.link1.angle = self.estimate_link1()
                print("Link 1 error", abs(robot.link1.angle - self.actual_link1))
                print("Estimated link", robot.link1.angle, "Actual link", self.actual_link1)
                self.prev_angle = self.actual_link1
        else:
            robot.angle = data.data[0]
            robot.angle = data.data[1]
            robot.angle = data.data[2]

    def estimate_link1(self):
        errors = []
        # idea: sample angles until we find a minimum
        prev_angle, error, prev_err, curr_angle = 0.0, 0.5, 0.0, 0.5
        while error > 0.02:
            robot.link1.angle = curr_angle
            pos = robot.update_effector_estimate()  # actual position
            print("Estimated postion", pos, robot.link1.angle, curr_angle)

            robot.link1.angle = self.actual_link1  # replace pos_d later with vision estimate
            pos_d = robot.update_effector_estimate()  # desired position
            print("Actual position", pos_d)
            error = np.sum((pos - pos_d) ** 2)

            error_d = (error - prev_err) / (curr_angle - prev_angle)  # derivative of error with rspct to angle
            
            # replacing old previous errors
            prev_err = error
            prev_angle = curr_angle

            curr_angle -= error / error_d
            errors.append(error)
        while curr_angle > np.pi or curr_angle < -np.pi:
            curr_angle -= 2 * np.pi if curr_angle > np.pi else -2 * np.pi

        return curr_angle


# run the code if the node is called
if __name__ == '__main__':
    fk = Link1Estimator()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down Link1Estimator")