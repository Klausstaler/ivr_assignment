#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError
from ivr_vision import ivr_vision, camera
from Link1Estimator import Link1Estimator
from forward_kinematics import robot
from sensor_msgs.msg import JointState
from copy import copy
import matplotlib.pyplot as plt

class image_converter:
  def __init__(self):
    self.test()
    rospy.init_node('image_processing', anonymous=True)
    self.image_pub1 = rospy.Publisher("image_topic1",Image, queue_size = 1)
    self.bridge = CvBridge()
    self._cam2_joint_locations_2d = np.repeat(None, 2 * 4).reshape(4, -1)
    self._joint_locations_2d = np.repeat(None, 2 * 4).reshape(4, -1)
    self._prev_angles = None
    self._joint_angles = np.array([0.0, 0.0, 0.0, 0.0])
    # comms
    self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw",Image,self.callback1)
    self.joint1_controller = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=3)
    self.joint2_controller = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=3)
    self.joint3_controller = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=3)
    self.joint4_controller = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=3)
    self.joint_angles_pub = rospy.Publisher("/robot/all_joints_estimate", Float64MultiArray, queue_size=3)
    self.cam2_joint1_location_2d_sub = rospy.Subscriber("/camera2/joint1_location_2d",Float64MultiArray,self.joint_locations_callback1)
    self.cam2_joint2_location_2d_sub = rospy.Subscriber("/camera2/joint2_location_2d",Float64MultiArray,self.joint_locations_callback2)
    self.cam2_joint3_location_2d_sub = rospy.Subscriber("/camera2/joint3_location_2d",Float64MultiArray,self.joint_locations_callback3)
    self.cam2_joint4_location_2d_sub = rospy.Subscriber("/camera2/joint4_location_2d",Float64MultiArray,self.joint_locations_callback4)

  def test(self):
    X = np.array([
        [[0, 0, 0], [0, 0, 2.5], [0, 0, 6]           , [0, 0, 9]],
        [[0, 0, 0], [0, 0, 2.5], [0, 3.5, 2.5]       , [0, 6, 2.5]],
        [[0, 0, 0], [0, 0, 2.5], [-1.75, -1.52, 5.12], [-2.81, -0.60, 7.78]],
        [[0, 0, 0], [0, 0, 2.5], [-2.31, .166, 5.13] , [-2.41, 1.56, 7.78]]
    ])
    Y = np.array([
        [0, 0, 0, 0],
        [0, 0, np.pi / 2, 0],
        [0         , np.pi / 6, -np.pi / 6, -np.pi / 4],
        [-np.pi / 4, np.pi / 6, -np.pi / 6, -np.pi / 4]
    ])
    errors = []
    predictions = []
    prev_estimate = -np.pi
    for theta1_truth in np.linspace(-np.pi, np.pi, num=30):
        angles = np.array([theta1_truth, np.pi / 6, -np.pi / 6, -np.pi / 4])
        # angles = np.array([theta1_truth, 0, 0, 1.3])
        _mat_1 = ivr_vision._transform(theta=np.pi/2, a=0.0, d=2.5, alpha=np.pi/2 , angle=angles[0])
        _mat_2 = ivr_vision._transform(theta=np.pi/2, a=0.0, d=0.0, alpha=np.pi/2 , angle=angles[1])
        _mat_3 = ivr_vision._transform(theta=0.0    , a=3.5, d=0.0, alpha=-np.pi/2, angle=angles[2])
        _mat_4 = ivr_vision._transform(theta=0.0    , a=3.0, d=0.0, alpha=0.0     , angle=angles[3])
        fk_joint_locs = np.array([
            [0.0, 0.0, 0.0],
            (_mat_1)[:-1, -1],
            (_mat_1 @ _mat_2 @ _mat_3)[:-1, -1],
            (_mat_1 @ _mat_2 @ _mat_3 @ _mat_4)[:-1, -1]
        ])
        estimated_angles, error = ivr_vision.fit_theta1(fk_joint_locs, prev_estimate)
        predictions.append([theta1_truth, estimated_angles[0]])
        errors.append([theta1_truth, error])
        prev_estimate = estimated_angles[0]
    predictions = np.array(predictions)
    errors = np.array(errors)
    plt.plot(predictions[:,0], predictions[:,1], c='gray')
    plt.xlabel(r'$\theta_1$')
    plt.xticks([-np.pi, np.pi], [r'$-\pi$', r'$\pi$'])
    plt.ylabel(r'$\hat{\theta_1}$')
    plt.yticks([-np.pi, np.pi], [r'$-\pi$', r'$\pi$'])
    plt.title(r'$\hat{\theta_1}$ as a function of $\theta_1$')
    plt.show()

  def joint_locations_callback1(self, data):
    self._cam2_joint_locations_2d[0] = np.array(data.data)
    self._joint_locations_callback(data)

  def joint_locations_callback2(self, data):
    self._cam2_joint_locations_2d[1] = np.array(data.data)
    self._joint_locations_callback(data)

  def joint_locations_callback3(self, data):
    self._cam2_joint_locations_2d[2] = np.array(data.data)
    self._joint_locations_callback(data)

  def joint_locations_callback4(self, data):
    self._cam2_joint_locations_2d[3] = np.array(data.data)
    self._joint_locations_callback(data)

  def _joint_locations_callback(self, data):
    if self._joint_locations_2d is None or None in self._cam2_joint_locations_2d:
        return
    Js = ivr_vision.combine_joint_locations(self._joint_locations_2d, self._cam2_joint_locations_2d)

    self._joint_angles, error = ivr_vision.fit_theta1(Js, self._joint_angles[0])

    self.joint_angles_pub.publish(Float64MultiArray(data=self._joint_angles))
    if (self._prev_angles is None or np.linalg.norm(self._prev_angles - self._joint_angles) > 0.2):
      print(f'angles: {self._joint_angles}')
      self._prev_angles = self._joint_angles

  def callback1(self,data):
    try:
      self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    ivr_vision.update_joint_locations(self.cv_image1, self._joint_locations_2d)
    im1=cv2.imshow('window1', self.cv_image1)
    cv2.waitKey(1)
    try:
      self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
    except CvBridgeError as e:
      print(e)

    time = rospy.get_time()
    # self._update_joint1(time)
    # self._update_joint2(time)
    # self._update_joint3(time)
    # self._update_joint4(time)

  def _update_joint1(self, t):
      new_state = np.pi * np.sin(np.pi / 15.0 * t)
      self.joint1_controller.publish(new_state)

  def _update_joint2(self, t):
      new_state = np.pi / 2.0 * np.sin(np.pi / 15.0 * t)
      self.joint2_controller.publish(new_state)

  def _update_joint3(self, t):
      new_state = np.pi / 2.0 * np.sin(np.pi / 18.0 * t)
      self.joint3_controller.publish(new_state)

  def _update_joint4(self, t):
      new_state = np.pi / 2.0 * np.sin(np.pi / 20.0 * t)
      self.joint4_controller.publish(new_state)

def main(args):
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
