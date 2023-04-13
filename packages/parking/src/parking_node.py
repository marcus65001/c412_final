#!/usr/bin/env python3

import rospy

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage, Range
from std_msgs.msg import Float32
from turbojpeg import TurboJPEG
import cv2
import numpy as np
import math
from duckietown_msgs.msg import Twist2DStamped, AprilTagDetection
from std_msgs.msg import String, Float32, Int32
from enum import Enum, auto
from collections import namedtuple

ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
ENGLISH = False
DEBUG = False
DEBUG_TEXT = True

#227

class State(Enum):
    GO_IN = auto()
    IP_LEFT = auto()
    IP_RIGHT = auto()
    LF = auto()
    DONE = auto()


class PD:
    def __init__(self, P=-0.049, D=0.004, clip=None):
        self.proportional = None
        self.P = P
        self.D = D
        self.last_error = 0
        self.last_time = rospy.get_time()
        self.disabled_value = None
        self.clip=clip

    def __repr__(self):
        return "<P={} D={} E={} DIS={}>".format(self.P, self.D, self.proportional, self.disabled_value)

    def get(self):
        # get the output of the PD
        if self.disabled_value is not None:
            return self.disabled_value
        if self.proportional is None:
            return 0
        # P Term
        P = self.proportional * self.P

        # D Term
        d_error = (self.proportional - self.last_error) / (
                rospy.get_time() - self.last_time) if self.last_error else 0
        self.last_error = self.proportional
        self.last_time = rospy.get_time()
        D = d_error * self.D
        if self.clip is not None:
            return np.clip(P+D,self.clip[0],self.clip[1])
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

class ConstantControl:
    def __init__(self, value):
        self.value=value

    def __repr__(self):
        return "<V={}>".format(self.value)

    def get(self):
        return self.value

class Controller:
    def __init__(self):
        self.cap_omega=6.5
        self.cap_v=0.7

    def get_velocity(self,node):
        return 0

    def get_omega(self,node):
        return 0

    def get_twist(self,node):
        return Twist2DStamped(v=self.get_velocity(node), omega=self.get_omega(node))


class ParkController(Controller):
    def __init__(self,velocity=0.46,ip_turn_omega=rospy.get_param("/ip_turn",3)):
        super().__init__()
        self.PD_omega = ConstantControl(0)
        self.PD_all = PD(1.0,0.02,clip=(0., 1.))
        # self.PD_all.set_disable(1.0)
        self.constant_v = ConstantControl(velocity)
        self.ip_turn_omega=ip_turn_omega

    def __repr__(self):
        return "<PD_omega={} PD_all={}>".format(self.PD_omega,self.PD_all)

    def get_omega(self,node):
        omega_cand=self.PD_omega.get()
        return max(min(omega_cand,self.cap_omega)*self.PD_all.get(),-self.cap_omega)

    def get_velocity(self,node):
        return max(0,min(self.constant_v.get()*self.PD_all.get(),self.cap_v))



class ParkingNode(DTROS):

    def __init__(self, node_name):
        super(ParkingNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.node_name = node_name
        self.veh = rospy.get_param("~veh")
        self.stall_number = rospy.get_param("/stall")

        self.jpeg = TurboJPEG()

        self.loginfo("Initialized")

        self.target_tag=227
        self.target_distance=rospy.get_param("/dist_go",0.4)
        self.state=State.GO_IN

        self.controller=ParkController()

        self.stall_numbers = {
            1: [207, State.IP_LEFT, 0.4],
            2: [226, State.IP_LEFT, 5.8],
            3: [228, State.IP_RIGHT, -2.1],
            4: [75, State.IP_RIGHT, -5.8],
        }

        self.twist = Twist2DStamped(v=0, omega=0)

        self.ip_turn_timer=None

        # Publishers & Subscribers
        self.sub_apriltag_info = rospy.Subscriber("~tag",
                                                  AprilTagDetection,
                                                  self.cb_apriltag_info,
                                                  queue_size=1)

        self.tof_sub = rospy.Subscriber("~tof_range",
                                        Range,
                                        self.cb_tof,
                                        queue_size=1)

        self.vel_pub = rospy.Publisher("/" + self.veh + "/car_cmd_switch_node/cmd",
                                       Twist2DStamped,
                                       queue_size=1)

        self.pub = rospy.Publisher("/" + self.veh + "/output/image/mask/compressed",
                                   CompressedImage,
                                   queue_size=1)

        # Wait a little while before sending motor commands
        rospy.Rate(0.20).sleep()

        # Shutdown hook
        rospy.on_shutdown(self.hook)

    def cb_apriltag_info(self, msg):
        if msg.tag_id:
            # Update detection info:
            self.tag_det = msg
            if self.tag_det.tag_id==self.target_tag:
                dist = self.tag_det.transform.translation.z
                self.loginfo(
                    "Target: {} - TDist: {} - Dist:{} - Cen:{}".format(self.target_tag, self.target_distance,dist,self.tag_det.center[0]))
                if self.state == State.GO_IN:
                    if dist<self.target_distance:
                        self.state=self.stall_numbers[self.stall_number][1]
                        self.target_tag=self.stall_numbers[self.stall_number][0]
                        self.loginfo("Stall: {} - Target: {} - State: {}".format(self.stall_number,self.target_tag,self.state))
                        self.target_distance=0.1
                        self.controller.constant_v.value=0
                        if self.state==State.IP_LEFT:
                            self.loginfo("Set Left")
                            self.controller.PD_omega=ConstantControl(self.controller.ip_turn_omega)
                            self.controller.PD_all.set_disable(1.0)
                            # self.ip_turn_timer=rospy.Timer(rospy.Duration(0.8),self.cb_ip_turn_timer)
                        elif self.state==State.IP_RIGHT:
                            self.loginfo("Set R")
                            self.controller.PD_omega=ConstantControl(-self.controller.ip_turn_omega)
                            self.controller.PD_all.set_disable(1.0)
                            # self.ip_turn_timer = rospy.Timer(rospy.Duration(0.8), self.cb_ip_turn_timer)
                        else:
                            self.controller.PD_all.set_disable(0)
                    else:
                        if self.controller.constant_v.value==0:
                            self.controller.constant_v.value=0.2
                        self.controller.PD_all.proportional=max(0,dist-self.target_distance)/self.target_distance
                elif self.state in {State.IP_LEFT,State.IP_RIGHT}:
                    self.controller.PD_all.set_disable(1.0)
                    # self.ip_turn_timer.shutdown()
                    self.state = State.DONE
                    self.loginfo("Found Tag")


    def cb_tof(self, msg):  # tof sensor, Range msg, in meters
        self.tof_det_range = msg.range if msg.range < msg.max_range else np.inf

        # if DEBUG_TEXT:
        #     self.loginfo("TOF DISTANCE: " + str(self.tof_det_range))


    def drive(self):
        rate = rospy.Rate(0.3)  # 8hz
        while not rospy.is_shutdown():
            if self.state == State.LF:
                self.loginfo(self.controller)
                tw = self.controller.get_twist(self)
                self.vel_pub.publish(tw)
            else:
                self.loginfo(self.controller)
                tw = self.controller.get_twist(self)
                self.vel_pub.publish(tw)
                rospy.sleep(rospy.Duration(0.5))
                self.twist.v = 0
                self.twist.omega = 0
                self.vel_pub.publish(self.twist)
            rate.sleep()

    def hook(self):
        print("SHUTTING DOWN")
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)
        for i in range(8):
            self.vel_pub.publish(self.twist)


if __name__ == "__main__":
    node = ParkingNode("parking_node")

    node.drive()
