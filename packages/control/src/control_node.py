#!/usr/bin/env python3

import rospy

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from turbojpeg import TurboJPEG
import cv2
import numpy as np
import math
from duckietown_msgs.msg import AprilTagDetectionArray, AprilTagDetection
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped
from duckietown_msgs.srv import ChangePattern, ChangePatternResponse
from std_msgs.msg import Int32, Bool
from enum import Enum, auto
from collections import namedtuple


ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
DEBUG = True


class State(Enum):
    LF = auto()
    LF_ENGLISH = auto()
    DUCK = auto()
    PARK = auto()
    MASK_RIGHT = auto()


# PD class
class PD:
    def __init__(self, P=-0.049, D=0.004):
        self.proportional = None
        self.P = P
        self.D = D
        self.last_error = 0
        self.last_time = rospy.get_time()
        self.disabled_value = None

    def __repr__(self):
        return "<P={} D={} E={} DIS={}>".format(self.P, self.D, self.proportional, self.disabled_value)

    def get(self):
        # get the output of the PD
        if self.disabled_value is not None:
            return self.disabled_value
        # P Term
        P = self.proportional * self.P

        # D Term
        d_error = (self.proportional - self.last_error) / (
                rospy.get_time() - self.last_time) if self.last_error else 0
        self.last_error = self.proportional
        self.last_time = rospy.get_time()
        D = d_error * self.D

        return P + D

    def reset(self):
        # reset the PD controller
        self.proportional = 0
        self.last_error = 0
        self.last_time = rospy.get_time()

    def set_disable(self, value):
        # set the PD controller to output a constant
        self.disabled_value = value
        self.reset()


class Controller:
    def __init__(self):
        self.cap_omega=8.0
        self.cap_v=0.8
    def get_velocity(self):
        return 0

    def get_omega(self):
        return 0

    def get_twist(self):
        return Twist2DStamped(v=self.get_velocity(), omega=self.get_omega())


class LF_Controller(Controller):
    def __init__(self,velocity):
        self.PD_omega = PD(-0.049, 0.004)
        self.velocity=velocity
    def get_omega(self):
        return max(min(self.PD_omega.get(),self.cap_omega),-self.cap_omega)

    def get_velocity(self):
        return max(0,min(self.velocity,self.cap_v))


class ControlNode(DTROS):
    def __init__(self, node_name):
        super(ControlNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.node_name = node_name
        self.veh = rospy.get_param("~veh")

        self.params = {}

        def get_param(name, default):
            # getting parameters from rosparam
            if name not in self.params:
                self.params[name] = rospy.get_param(name, default)
            return self.params[name]

        self.jpeg = TurboJPEG()

        self.loginfo("Initialized")

        # Properties
        self.state = State.LF

        # Publishers & Subscribers
        self.pub = rospy.Publisher("/" + self.veh + "/output/image/mask/compressed",
                                   CompressedImage,
                                   queue_size=1)
        self.sub = rospy.Subscriber("/" + self.veh + "/camera_node/image/compressed",
                                    CompressedImage,
                                    self.callback,
                                    queue_size=1,
                                    buff_size="20MB")
        self.vel_pub = rospy.Publisher("/" + self.veh + "/car_cmd_switch_node/cmd",
                                       Twist2DStamped,
                                       queue_size=1)

        # Shutdown hook
        rospy.on_shutdown(self.hook)

    # Calculates the midpoint of the contoured object
    def midpoint(self, x, y, w, h):
        mid_x = int(x + (((x + w) - x) / 2))
        mid_y = int(y + (((y + h) - y)))
        return (mid_x, mid_y)

    def callback(self, msg):
        img = self.jpeg.decode(msg.data)
        # Part for Lane Following Detection
        crop = img[300:-1, :, :]
        crop_width = crop.shape[1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])
        crop = cv2.bitwise_and(crop, crop, mask=mask)
        contours, hierarchy = cv2.findContours(mask,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)

        # Search for lane in front
        max_area = 20
        max_idx = -1
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_idx = i
                max_area = area

        if max_idx != -1:
            M = cv2.moments(contours[max_idx])
            try:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                self.pd_omega.proportional = cx - int(crop_width / 2) + self.lf_offset
                if DEBUG:
                    cv2.drawContours(crop, contours, max_idx, (0, 255, 0), 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
            except:
                pass
        else:
            self.pd_omega.proportional = None

        if DEBUG:
            rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(crop))
            self.pub.publish(rect_img_msg)

    def drive(self):
        self.twist=self.controller.get_twist()
        self.vel_pub.publish(self.twist)

    def hook(self):
        print("SHUTTING DOWN")
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)
        for i in range(8):
            self.vel_pub.publish(self.twist)


if __name__ == "__main__":
    node = ControlNode("control_node")
    rate = rospy.Rate(8)  # 8hz
    while not rospy.is_shutdown():
        node.drive()
        rate.sleep()