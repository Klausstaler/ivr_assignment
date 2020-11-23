import rospy
import numpy as np
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
class Tester:

    def __init__(self):
        self.optimal_joint_sub = rospy.Subscriber("/target/joint_states", JointState, self.optimal_joint_cb)
        self.joints = None

    def optimal_joint_cb(self, ):