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
    rospy.init_node('image_processing', anonymous=True)
    self.image_pub2 = rospy.Publisher("image_topic2",Image, queue_size = 1)
    self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw",Image,self.callback2)
    self.joint1_location_2d_pub = rospy.Publisher("/camera2/joint1_location_2d",Float64MultiArray, queue_size = 1)
    self.joint2_location_2d_pub = rospy.Publisher("/camera2/joint2_location_2d",Float64MultiArray, queue_size = 1)
    self.joint3_location_2d_pub = rospy.Publisher("/camera2/joint3_location_2d",Float64MultiArray, queue_size = 1)
    self.joint4_location_2d_pub = rospy.Publisher("/camera2/joint4_location_2d",Float64MultiArray, queue_size = 1)
    self.bridge = CvBridge()


  # Recieve data, process it, and publish
  def callback2(self,data):
    # Recieve the image
    try:
      self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    self._joint_locations = ivr_vision.detect_joint_locations(self.cv_image2)
    self.joint1_location_2d_pub.publish(Float64MultiArray(data=self._joint_locations[0]))
    self.joint2_location_2d_pub.publish(Float64MultiArray(data=self._joint_locations[1]))
    self.joint3_location_2d_pub.publish(Float64MultiArray(data=self._joint_locations[2]))
    self.joint4_location_2d_pub.publish(Float64MultiArray(data=self._joint_locations[3]))

    # Uncomment if you want to save the image
    #cv2.imwrite('image_copy.png', cv_image)
    im2=cv2.imshow('window2', self.cv_image2)
    cv2.waitKey(1)

    # Publish the results
    try: 
      self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))
    except CvBridgeError as e:
      print(e)

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


