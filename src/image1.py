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

class image_converter:

  # Defines publisher and subscriber
  def __init__(self):
    # initialize the node named image_processing
    rospy.init_node('image_processing', anonymous=True)
    # initialize a publisher to send images from camera1 to a topic named image_topic1
    self.image_pub1 = rospy.Publisher("image_topic1",Image, queue_size = 1)
    # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
    self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw",Image,self.callback1)
    self.joint1_controller = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=3)
    self.joint2_controller = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=3)
    self.joint3_controller = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=3)
    self.joint4_controller = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=3)
    self.joint_angles_pub = rospy.Publisher("/robot/joint_angles", Float64MultiArray, queue_size=3)
    self.cam2_joint1_location_2d_sub = rospy.Subscriber("/camera2/joint1_location_2d",Float64MultiArray,self.joint_locations_callback1)
    self.cam2_joint2_location_2d_sub = rospy.Subscriber("/camera2/joint2_location_2d",Float64MultiArray,self.joint_locations_callback2)
    self.cam2_joint3_location_2d_sub = rospy.Subscriber("/camera2/joint3_location_2d",Float64MultiArray,self.joint_locations_callback3)
    self.cam2_joint4_location_2d_sub = rospy.Subscriber("/camera2/joint4_location_2d",Float64MultiArray,self.joint_locations_callback4)
    self._cam2_joint_locations_2d = np.repeat(None, 2 * 4).reshape(4, -1)
    self._joint_locations_2d = None
    self._camera = camera(np.array([18, 0, 0]))
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()

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

  # receive 2D joint locations from camera2, combine them into 3D
  def _joint_locations_callback(self, data):
    if self._joint_locations_2d is None or None in self._cam2_joint_locations_2d:
        return
    joint_locations_3d = ivr_vision.combine_joint_locations(self._joint_locations_2d, self._cam2_joint_locations_2d)
    self._joint_angles = ivr_vision.compute_joint_angles(joint_locations_3d)
    message = Float64MultiArray()
    message.data = self._joint_angles
    self.joint_angles_pub.publish(message)

  # Recieve data from camera 1, process it, and publish
  def callback1(self,data):
    # Recieve the image
    try:
      self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    self._joint_locations_2d = ivr_vision.detect_joint_locations(self.cv_image1)

    # Uncomment if you want to save the image
    #cv2.imwrite('image_copy.png', cv_image)

    im1=cv2.imshow('window1', self.cv_image1)
    cv2.waitKey(1)
    # Publish the results
    try:
      self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
    except CvBridgeError as e:
      print(e)

    time = rospy.get_time()
#    self._update_joint2(time)
#    self._update_joint3(time)
#    self._update_joint4(time)

  def _update_joint2(self, t):
      new_state = np.pi / 2.0 * np.sin(np.pi / 15.0 * t)
      self.joint2_controller.publish(new_state)

  def _update_joint3(self, t):
      new_state = np.pi / 2.0 * np.sin(np.pi / 18.0 * t)
      self.joint3_controller.publish(new_state)

  def _update_joint4(self, t):
      new_state = np.pi / 2.0 * np.sin(np.pi / 20.0 * t)
      self.joint4_controller.publish(new_state)

# call the class
def main(args):
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
