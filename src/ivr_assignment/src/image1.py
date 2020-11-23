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
    
    self._camera = camera(np.array([18, 0, 0]))
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()


  # Recieve data from camera 1, process it, and publish
  def callback1(self,data):
    # Recieve the image
    try:
      self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    
    self._detect_joint_angles(self.cv_image1)

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
    self._update_joint2(time)
    self._update_joint3(time)
    self._update_joint4(time)

  def _detect_joint_angles(self, image):
    yellow = ivr_vision.detect_blob(image, ivr_vision.YELLOW_RANGE)
    blue = ivr_vision.detect_blob(image, ivr_vision.BLUE_RANGE)
    green = ivr_vision.detect_blob(image, ivr_vision.GREEN_RANGE)
    red = ivr_vision.detect_blob(image, ivr_vision.RED_RANGE)
    if ivr_vision.DEBUG:
        r = 5
        # draw dots in the centers to check that this works
        cv2.circle(image, tuple(yellow), r, ivr_vision.invert(ivr_vision.YELLOW_RANGE[1]), -1)
        cv2.circle(image, tuple(blue), r, ivr_vision.invert(ivr_vision.BLUE_RANGE[1]), -1)
        cv2.circle(image, tuple(green), r, ivr_vision.invert(ivr_vision.GREEN_RANGE[1]), -1)
        cv2.circle(image, tuple(red), r, ivr_vision.invert(ivr_vision.RED_RANGE[1]), -1)
    # find equations for line that satisfy blob locations

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


