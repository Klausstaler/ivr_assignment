#!/usr/bin/env python3

import rospy
import numpy as np
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64, Float64MultiArray
from sensor_msgs.msg import JointState
from jacobian_symp import calculate_jacobian
class Link:
    def __init__(self, theta=0.0, a=0.0, d=0.0, alpha=.0):
        self.angle, self.theta, self.a, self.d, self.alpha = 0, theta, a, d, alpha

    def calc_trans(self):
        x_rot, x_trans, z_trans, z_rot = np.eye(4), np.eye(4), np.eye(4), np.eye(4)
        x_rot[:-1, :-1] = R.from_euler("xyz", [self.alpha, 0, 0]).as_matrix()
        x_trans[0, -1] = self.a
        z_rot[:-1, :-1] = R.from_euler("xyz", [0, 0, self.theta + self.angle]).as_matrix()
        z_trans[-2, -1] = self.d

        return z_rot @ z_trans @ x_trans @ x_rot
class Robot:
    def __init__(self, link1, link2, link3, link4):
        self.link1, self.link2, self.link3, self.link4 = link1, link2, link3, link4
        self.jac = calculate_jacobian(self)

    def update_effector_estimate(self):
        link1_mat = self.link1.calc_trans()
        link2_mat = self.link2.calc_trans()
        link3_mat = self.link3.calc_trans()
        link4_mat = self.link4.calc_trans()
        joint_to_pos = (link1_mat @ link2_mat @ link3_mat @ link4_mat)[:-1, -1]
        return joint_to_pos

link1 = Link(theta=np.pi/2, d=2.5, alpha=np.pi/2)
link2 = Link(theta=np.pi/2, alpha=np.pi/2)
link3 = Link(a=3.5, alpha=-np.pi/2)
link4 = Link(a=3)
robot = Robot(link1, link2, link3, link4)
class KinematicsCalculator:

    def __init__(self):
        rospy.init_node('KinematicsCalculator', anonymous=True)
        # initialize a publisher to send images from camera1 to a topic named image_topic1
        self.x_pub = rospy.Publisher("forward_kinematics/x", Float64, queue_size=1)
        self.y_pub = rospy.Publisher("forward_kinematics/y", Float64, queue_size=1)
        self.z_pub = rospy.Publisher("forward_kinematics/z", Float64, queue_size=1)
        # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
        self.target_pos = np.array([0.0, 0., 0.])
        self.time_previous_step = np.array([0.0])

        #self.link_sub = rospy.Subscriber("/robot/joint_states", JointState, self.links_cb)
        self.link_sub = rospy.Subscriber("/robot/joints_estimate", Float64MultiArray, self.links_cb)
        """self.target_x_sub = rospy.Subscriber("/target/x_position_controller/command", Float64, self.target_x_cb)
        self.target_y_sub = rospy.Subscriber("/target/y_position_controller/command", Float64, self.target_y_cb)
        self.target_z_sub = rospy.Subscriber("/target/z_position_controller/command", Float64, self.target_z_cb)"""
        self.target_sub = rospy.Subscriber("/robot/target_location_estimate", Float64MultiArray, self.target_cb)

        self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=1)
        self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=1)
        self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=1)
        self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=1)
        # initialize error and derivative of error for trajectory tracking
        self.error = np.array([0.0, 0.0, 0.0], dtype='float64')
        self.error_d = np.array([0.0, 0.0, 0.0], dtype='float64')

    def link1_cb(self, data):
        robot.link1.angle = data.data
        #self.update_effector_estimate()
    def link2_cb(self, data):
        robot.link2.angle = data.data
        #self.update_effector_estimate()
    def link3_cb(self, data):
        robot.link3.angle = data.data
        #self.update_effector_estimate()
    def link4_cb(self, data):
        robot.link4.angle = data.data
        #self.update_effector_estimate()
    def links_cb(self, data):
        if type(data) == JointState:
            angles = data.position
            robot.link1.angle = angles[0]
            robot.link2.angle = angles[1]
            robot.link3.angle = angles[2]
            robot.link4.angle = angles[3]
            pass
        else:
            robot.link2.angle = data.data[0]
            robot.link3.angle = data.data[1]
            robot.link4.angle = data.data[2]


    def target_cb(self, data):
        data = data.data
        self.target_pos[0], self.target_pos[1], self.target_pos[2] = data[0], data[1], data[2]
        self.control_closed()
    def target_x_cb(self, data):
        self.target_pos[0] = data.data
        #self.control_closed()
    def target_y_cb(self, data):
        self.target_pos[1] = data.data
        #self.control_closed()
    def target_z_cb(self, data):
        self.target_pos[2] = data.data
        self.control_closed()

    def control_closed(self):
        # P gain
        p, d = 10, 0.4
        K_p = np.array([[p, 0.0, 0.0], [0.0, p, 0.0], [0.0, 0.0, p]])
        # D gain
        K_d = np.array([[d, 0, 0], [0, d, 0], [0, 0, d]])
        #K_d = np.zeros(shape=(3,3))
        damper = 1.3
        # estimate time step
        cur_time = np.array([rospy.get_time()])
        dt = cur_time - self.time_previous_step


        if dt == 0:
            return

        # robot end-effector position
        pos = robot.update_effector_estimate()
        # desired position
        pos_d = self.target_pos
        # estimate derivative of error
        self.error_d = ((pos_d - pos) - self.error) / dt
        # estimate error
        self.error = pos_d - pos
        if self.time_previous_step == 0:
            self.time_previous_step = cur_time
            return
        self.time_previous_step = cur_time

        q = np.array([robot.link1.angle, robot.link2.angle, robot.link3.angle, robot.link4.angle]) # by removing link 1 we fixed it
        #q = q[1:]
        q[0] = 0
        jacobian = robot.jac(0, q[1], q[2], q[3])
        jacobian[:, 0] = 0
        #jacobian = self.calc_jacobian()
        J_inv = jacobian.T @ np.linalg.inv(jacobian@jacobian.T + np.eye(jacobian.shape[0])*damper)

        #J_inv = np.linalg.pinv(jacobian)  # calculating the pseudo inverse of Jacobian
        #print(K_d.shape)
        err = (K_d@(self.error_d.reshape(3,1)) + K_p@(self.error.reshape(3,1)))
        dq_d = J_inv@(K_d@(self.error_d.reshape(3,1)) + K_p@(self.error.reshape(3,1)))  # control input (angular velocity of joints)
        q_d = q + (dt * dq_d.flatten())  # control input (angular position of joints)

        for i, num in enumerate(q_d):
            #while q_d[i] > np.pi:
            #    q_d[i] -= 2 * np.pi
            #while q_d[i] < -np.pi:
            #    q_d[i] += 2 * np.pi
            q_d[i] = min(q_d[i], np.pi / 2)
            q_d[i] = max(q_d[i], -np.pi / 2)


        if not np.any(np.isnan(q_d)):
            print(q_d)
            print("Desired position", pos_d)
            print("Current position", pos)
            self.robot_joint2_pub.publish(q_d[1])
            self.robot_joint3_pub.publish(q_d[2])
            self.robot_joint4_pub.publish(q_d[3])
            #self.robot_joint4_pub.publish(q_d[2])
        return q_d

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
