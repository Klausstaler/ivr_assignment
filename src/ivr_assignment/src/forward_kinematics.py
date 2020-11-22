#!/usr/bin/env python3

import rospy
import numpy as np
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64

class KinematicsCalculator:

    def __init__(self):
        rospy.init_node('KinematicsCalculator', anonymous=True)
        # initialize a publisher to send images from camera1 to a topic named image_topic1
        self.x_pub = rospy.Publisher("forward_kinematics/x", Float64, queue_size=1)
        self.y_pub = rospy.Publisher("forward_kinematics/y", Float64, queue_size=1)
        self.z_pub = rospy.Publisher("forward_kinematics/z", Float64, queue_size=1)
        # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
        self.link1_angle, self.link2_angle, self.link3_angle, self.link4_angle = 0, 0, 0, 0
        self.link1_sub = rospy.Subscriber("/robot/joint1_position_controller/command", Float64, self.link1_cb)
        self.link2_sub = rospy.Subscriber("/robot/joint2_position_controller/command", Float64, self.link2_cb)
        self.link3_sub = rospy.Subscriber("/robot/joint3_position_controller/command", Float64, self.link2_cb)
        self.link4_sub = rospy.Subscriber("/robot/joint4_position_controller/command", Float64, self.link2_cb)

    def link1_cb(self, data):
        self.link1_angle = data.data
        self.update_effector_estimate()
    def link2_cb(self, data):
        self.link2_angle = data.data
        self.update_effector_estimate()
    def link3_cb(self, data):
        self.link3_angle = data.data
        self.update_effector_estimate()
    def link4_cb(self, data):
        self.link4_angle = data.data
        self.update_effector_estimate()

    def update_effector_estimate(self):
        link1_mat = self.invert_affine_mat(self.calc_trans(self.link1_angle + np.pi/2, 2.5, alpha=np.pi/2))
        link2_mat = self.invert_affine_mat(self.calc_trans(self.link2_angle + np.pi/2, 0, 0, alpha=np.pi/2))
        link3_mat = self.invert_affine_mat(self.calc_trans(self.link3_angle, 0, 3.5, -np.pi/2))
        link4_mat = self.invert_affine_mat(self.calc_trans(self.link4_angle, 0, 3, np.pi/2))
        # sooo uhmm I went from base from to end effector frame. But we want from end effector frame to base frame (lul)
        joint_to_pos = (link1_mat@link2_mat@link3_mat@link4_mat)[:-1, -1]
        self.x_pub.publish(joint_to_pos[0])
        self.y_pub.publish(joint_to_pos[1])
        self.z_pub.publish(joint_to_pos[2])
        #self.x_pub.publish(joint_to_pos[2])
        #self.y_pub.publish(joint_to_pos[1])
        #self.z_pub.publish(joint_to_pos[0])


    def invert_affine_mat(self, mat):
        inverse = np.eye(4)
        rot_inv = mat[:-1, :-1].T # transpose of rotation matrix is its' inverse
        trans_inv = -rot_inv@mat[:-1, -1]
        inverse[:-1, :-1] = rot_inv
        inverse[:-1, -1] = trans_inv
        return inverse
    def calc_trans(self, theta=0.0, d=0.0, a=0.0, alpha=0.0):
        x_rot, x_trans, z_trans, z_rot = np.eye(4), np.eye(4), np.eye(4), np.eye(4)
        x_rot[:-1, :-1] = R.from_euler("xyz", [alpha, 0, 0]).as_matrix()
        x_trans[0, -1] = a
        z_rot[:-1, :-1] = R.from_euler("xyz", [0, 0, theta]).as_matrix()
        z_trans[-2, -1] = d

        return z_rot @ z_trans @ x_trans @ x_rot

    def link_length(self, joint_start, joint_end, real_length):
        dist = np.sum((joint_start - joint_end) ** 2)
        return real_length / np.sqrt(dist)  # link length in pixels

# run the code if the node is called
if __name__ == '__main__':
    fk = KinematicsCalculator()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down KinematicsCalculator")