#!/usr/bin/env python3

import rospy
import numpy as np
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64
import time

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
        self.target_pos = np.array([0, 0, 0])
        self.time_previous_step = rospy.get_time()
        self.link1_sub = rospy.Subscriber("/robot/joint1_position_controller/command", Float64, self.link1_cb)
        self.link2_sub = rospy.Subscriber("/robot/joint2_position_controller/command", Float64, self.link2_cb)
        self.link3_sub = rospy.Subscriber("/robot/joint3_position_controller/command", Float64, self.link3_cb)
        self.link4_sub = rospy.Subscriber("/robot/joint4_position_controller/command", Float64, self.link4_cb)
        self.target_x_sub =  rospy.Subscriber("/target/x_position_controller/command", Float64, self.target_x_cb)
        self.target_y_sub = rospy.Subscriber("/target/y_position_controller/command", Float64, self.target_y_cb)
        self.target_z_sub = rospy.Subscriber("/target/z_position_controller/command", Float64, self.target_z_cb)

        #self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
        #self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
        #self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)
        # initialize error and derivative of error for trajectory tracking
        self.error = np.array([0.0, 0.0, 0.0], dtype='float64')
        self.error_d = np.array([0.0, 0.0, 0.0], dtype='float64')

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
    def target_x_cb(self, data):
        self.target_pos[0] = data.data
        #self.control_closed()
    def target_y_cb(self, data):
        self.target_pos[1] = data.data
        #self.control_closed()
    def target_z_cb(self, data):
        self.target_pos[2] = data.data
        #self.control_closed()

    def update_effector_estimate(self):
        link1_mat = self.calc_trans(self.link1.angle + self.link1.theta, d=self.link1.d, alpha=self.link2.alpha)
        link2_mat = self.calc_trans(self.link2.angle + self.link2.theta, alpha=self.link2.alpha)
        link3_mat = self.calc_trans(self.link3.angle, a=self.link3.a, alpha=self.link3.alpha)
        link4_mat = self.calc_trans(self.link4.angle, a=self.link4.a)
        joint_to_pos = (link1_mat@link2_mat@link3_mat@link4_mat)[:-1, -1]
        self.x_pub.publish(joint_to_pos[0])
        self.y_pub.publish(joint_to_pos[1])
        self.z_pub.publish(joint_to_pos[2])
        return joint_to_pos

    def calc_trans(self, theta=0.0, d=0.0, a=0.0, alpha=0.0):
        x_rot, x_trans, z_trans, z_rot = np.eye(4), np.eye(4), np.eye(4), np.eye(4)
        x_rot[:-1, :-1] = R.from_euler("xyz", [alpha, 0, 0]).as_matrix()
        x_trans[0, -1] = a
        z_rot[:-1, :-1] = R.from_euler("xyz", [0, 0, theta]).as_matrix()
        z_trans[-2, -1] = d

        return z_rot @ z_trans @ x_trans @ x_rot

    def control_closed(self):
        # P gain
        K_p = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
        # D gain
        K_d = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
        # estimate time step
        cur_time = np.array([rospy.get_time()])
        dt = cur_time - self.time_previous_step
        if dt < 0.1:
            return
        self.time_previous_step = cur_time
        # robot end-effector position
        pos = self.update_effector_estimate()
        # desired position
        pos_d = self.target_pos
        # estimate derivative of error
        self.error_d = ((pos_d - pos) - self.error) / dt
        # estimate error
        self.error = pos_d - pos
        q = np.array([self.link1.angle, self.link2.angle, self.link3.angle, self.link4.angle])
        J_inv = np.linalg.pinv(self.calc_jacobian())  # calculating the pseudo inverse of Jacobian
        dq_d = np.dot(J_inv, (np.dot(K_d, self.error_d.transpose()) + np.dot(K_p,
                                                                             self.error.transpose())))  # control input (angular velocity of joints)
        q_d = q + (dt * dq_d)  # control input (angular position of joints)
        if not np.any(np.isnan(q_d)):
            #print(q_d)
            self.robot_joint2_pub.publish(q_d[1])
            self.robot_joint3_pub.publish(q_d[2])
            self.robot_joint4_pub.publish(q_d[3])
        return q_d

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