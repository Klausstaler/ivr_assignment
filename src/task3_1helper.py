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


joint_angles = np.array([[0.3, 0.3, 0.3, 0.3],
                        [-0.6, -0.6, 0.6, 0.6],
                        [1.0,-0.6, 0.6, -1.0],
                        [1.0, 0.6, -0.6, -1.0],
                        [-0.5, 0.3, 0.1, -0.4],
                        [-.2, -.2, -.2, 0.7],
                        [.8, .8, .8, .8],
                        [.4, -.3, .4, .2],
                        [.1, -.1, .1, -.1],
                        [.6, .5, -.5, .5]])

class image_converter:

  # Defines publisher and subscriber
  def __init__(self):
    rospy.init_node('3_1helper', anonymous=True)
    self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=1)
    self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=1)
    self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=1)
    self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=1)
    for joint_angle in joint_angles:
        print("Publishing joint angles", joint_angle)
        input("Publish?")
        self.robot_joint1_pub.publish(joint_angle[0])
        self.robot_joint2_pub.publish(joint_angle[1])
        self.robot_joint3_pub.publish(joint_angle[2])
        self.robot_joint4_pub.publish(joint_angle[3])
        input("Now next input")

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
