#!/usr/bin/env python3

import rospy
import numpy as np
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64

class Link:
    def __init__(self, theta=0.0, a=0.0, d=0.0, alpha=.0):
        self.angle, self.theta, self.a, self.d, self.alpha = 0, theta, a, d, alpha
class KinematicsCalculator:

    def __init__(self):
        rospy.init_node('KinematicsCalculator', anonymous=True)
        # initialize a publisher to send images from camera1 to a topic named image_topic1
        self.x_pub = rospy.Publisher("forward_kinematics/x", Float64, queue_size=1)
        self.y_pub = rospy.Publisher("forward_kinematics/y", Float64, queue_size=1)
        self.z_pub = rospy.Publisher("forward_kinematics/z", Float64, queue_size=1)
        # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
        self.link1, self.link2, self.link3, self.link4 = Link(theta=np.pi/2, d=2.5, alpha=np.pi/2), \
                                                         Link(theta=np.pi/2, alpha=np.pi/2), \
                                                         Link(a=3.5, alpha=-np.pi/2), \
                                                         Link(a=3)
        self.link1_sub = rospy.Subscriber("/robot/joint1_position_controller/command", Float64, self.link1_cb)
        self.link2_sub = rospy.Subscriber("/robot/joint2_position_controller/command", Float64, self.link2_cb)
        self.link3_sub = rospy.Subscriber("/robot/joint3_position_controller/command", Float64, self.link3_cb)
        self.link4_sub = rospy.Subscriber("/robot/joint4_position_controller/command", Float64, self.link4_cb)

    def link1_cb(self, data):
        self.link1.angle = data.data
        self.update_effector_estimate()
    def link2_cb(self, data):
        self.link2.angle = data.data
        self.update_effector_estimate()
    def link3_cb(self, data):
        self.link3.angle = data.data
        self.update_effector_estimate()
    def link4_cb(self, data):
        self.link4.angle = data.data
        self.update_effector_estimate()

    def update_effector_estimate(self):
        link1_mat = self.calc_trans(self.link1.angle + self.link1.theta, d=self.link1.d, alpha=self.link2.alpha)
        link2_mat = self.calc_trans(self.link2.angle + self.link2.theta, alpha=self.link2.alpha)
        link3_mat = self.calc_trans(self.link3.angle, a=self.link3.a, alpha=self.link3.alpha)
        link4_mat = self.calc_trans(self.link4.angle, a=self.link4.a)
        joint_to_pos = (link1_mat@link2_mat@link3_mat@link4_mat)[:-1, -1]
        self.x_pub.publish(joint_to_pos[0])
        self.y_pub.publish(joint_to_pos[1])
        self.z_pub.publish(joint_to_pos[2])

    def calc_trans(self, theta=0.0, d=0.0, a=0.0, alpha=0.0):
        x_rot, x_trans, z_trans, z_rot = np.eye(4), np.eye(4), np.eye(4), np.eye(4)
        x_rot[:-1, :-1] = R.from_euler("xyz", [alpha, 0, 0]).as_matrix()
        x_trans[0, -1] = a
        z_rot[:-1, :-1] = R.from_euler("xyz", [0, 0, theta]).as_matrix()
        z_trans[-2, -1] = d

        return z_rot @ z_trans @ x_trans @ x_rot


    def calc_jacobian(self):
        jacobian = np.zeros((3,4))

        # this is gonna look very nasty

        # angles of first joint
        sin_t1, cos_t1 = self.getsin_cos(self.link1.theta + self.link1.angle)
        sin_a1, cos_a1 = self.getsin_cos(self.link1.alpha)
        # second joint
        sin_t2, cos_t2 = self.getsin_cos(self.link2.angle + self.link2.theta)
        sin_a2, cos_a2 = self.getsin_cos(self.link2.alpha)
        # third joint
        sin_t3, cos_t3 = self.getsin_cos(self.link3.angle + self.link3.theta)
        sin_a3, cos_a3 = self.getsin_cos(self.link3.alpha)
        # fourth joint
        sin_t4, cos_t4 = self.getsin_cos(self.link4.angle + self.link4.theta)

        # now we get the derivate of x with respect to all angles

        # x/dt1 is 0
        # so let's go for x/dt2
        jacobian[0, 1] = self.link1.d*(sin_a2*cos_t2*(cos_t3*cos_t4-sin_t3*sin_t4) -
                        sin_a2*sin_t2*(sin_t4*cos_a3*cos_t3 + cos_t4*cos_a3*sin_t3))
        # now x/dt3
        jacobian[0, 2] = -self.link3.a*sin_t3*cos_t4 - self.link3.a*sin_t4*cos_t3 + \
                         self.link1.d*((-sin_t3*cos_t4 - sin_t4*cos_t3)*sin_a2*sin_t2 +
                         (sin_a3*sin_t3*sin_t4 - sin_a3*cos_t3*cos_t4)*cos_a2 + (-sin_t3*sin_t4*cos_a3 +
                         cos_a3*cos_t3*cos_t4)*sin_a2*cos_t2)
        # now x/dt4
        jacobian[0, 3] = -self.link3.a*sin_t3*cos_t4 - self.link3.a*sin_t4*cos_t3 - \
                         self.link4.a*sin_t4 + self.link1.d*((-sin_t3*cos_t4 - sin_t4*cos_t3)*sin_a2*sin_t2 +
                         (sin_a3*sin_t3*sin_t4 - sin_a3*cos_t3*cos_t4)*cos_a2 + (-sin_t3*sin_t4*cos_a3 +
                         cos_a3*cos_t3*cos_t4)*sin_a2*cos_t2)
        # now y with respect to all angles
        # y/dt1 is zero again
        # so we start with y/dt2
        jacobian[1, 1] = self.link1.d*((sin_t3*cos_t4 + sin_t4*cos_t3)*sin_a2*cos_t2 -
                         (cos_a3*sin_t3*sin_t4 - cos_a3*cos_t3*cos_t4)*sin_a2*sin_t2)
        # now y/dt3
        jacobian[1, 2] = -self.link3.a*sin_t3*sin_t4 + self.link3.a*cos_t4*cos_t3 + \
                         self.link1.d*((-sin_t3*sin_t4 + cos_t4*cos_t3)*sin_a2*sin_t2 +
                         (-sin_a3*sin_t3*cos_t4 - sin_a3*cos_t3*sin_t4)*cos_a2 + (sin_t3*cos_t4*cos_a3 +
                         cos_a3*cos_t3*sin_t4)*sin_a2*cos_t2)
        # now y/dt4
        jacobian[1, 3] = -self.link3.a*sin_t3*sin_t4 + self.link3.a*cos_t4*cos_t3 + self.link4.a*cos_t4 \
                         + self.link1.d*((-sin_t3*sin_t4 + cos_t4*cos_t3)*sin_a2*sin_t2 +
                         (-sin_a3*sin_t3*cos_t4 - sin_a3*cos_t3*sin_t4)*cos_a2 + (sin_t3*cos_t4*cos_a3 +
                         cos_a3*cos_t3*sin_t4)*sin_a2*cos_t2)

        # now z with respect to all angles
        # again z/dt1 is zero
        # now z/dt2
        jacobian[2, 1] = self.link1.d*sin_a2*sin_a3*sin_t2
        # rest is zero
        return jacobian


    def getsin_cos(self, angle):
        return np.sin(angle), np.cos(angle)
    def invert_affine_mat(self, mat):
        inverse = np.eye(4)
        rot_inv = mat[:-1, :-1].T # transpose of rotation matrix is its' inverse
        trans_inv = -rot_inv@mat[:-1, -1]
        inverse[:-1, :-1] = rot_inv
        inverse[:-1, -1] = trans_inv
        return inverse

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