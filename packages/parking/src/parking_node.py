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
        self.stall_number = rospy.get_param("/stall",1)

        self.jpeg = TurboJPEG()

        self.loginfo("Initialized")

        self.target_tag=227
        self.target_distance=rospy.get_param("/dist_go",0.4)
        self.state=State.GO_IN

        self.ip_turn_omega=rospy.get_param("/ip_turn",3.0)
        self.velocity=rospy.get_param("/velocity",0.3)

        self.tof_distance=np.inf

        self.stall_numbers = {
            1: [207, State.IP_LEFT, 0.4],
            2: [226, State.IP_LEFT, 5.8],
            3: [228, State.IP_RIGHT, -2.1],
            4: [75, State.IP_RIGHT, -5.8],
        }

        self.twist = Twist2DStamped(v=0, omega=0)
        self.rate = rospy.Rate(0.3)

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
                if DEBUG:
                    self.loginfo(
                        "Target: {} - TDist: {} - Dist:{} - Cen:{}".format(self.target_tag, self.target_distance,dist,self.tag_det.center[0]))
                if self.state == State.GO_IN:
                    if dist<self.target_distance:
                        self.state=self.stall_numbers[self.stall_number][1]
                        self.target_tag=self.stall_numbers[self.stall_number][0]
                        if DEBUG:
                            self.loginfo("Stall: {} - Target: {} - State: {}".format(self.stall_number,self.target_tag,self.state))
                        self.twist.v=0
                        if self.state==State.IP_LEFT:
                            if DEBUG:
                                self.loginfo("Set Left")
                            self.twist.omega=self.ip_turn_omega
                        elif self.state==State.IP_RIGHT:
                            if DEBUG:
                                self.loginfo("Set Right")
                            self.twist.omega=-self.ip_turn_omega
                    else:
                        self.twist.v=self.velocity
                elif self.state in {State.IP_LEFT,State.IP_RIGHT}:
                    self.twist.omega = 0
                    self.rate=rospy.Rate(8)
                    self.state = State.LF
                    if DEBUG:
                        self.loginfo("Found Tag")
            else:
                if DEBUG:
                    self.loginfo("Lost Target")
                self.twist.v = 0
                self.twist.omega = 0


    def callback(self, msg):

        if self.state!=State.LF:
            return

        img = self.jpeg.decode(msg.data)

        # add_patch
        m_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        m_mask[:, :-300] = 1
        img = cv2.bitwise_and(img, img, mask=m_mask)
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
                self.proportional = cx - int(crop_width / 2) + self.offset
                if DEBUG:
                    cv2.drawContours(crop, contours, max_idx, (0, 255, 0), 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
            except:
                pass
        else:
            self.proportional = None


    def cb_tof(self, msg):  # tof sensor, Range msg, in meters
        self.tof_det_range = msg.range if msg.range < msg.max_range else np.inf


    def drive(self):
        while not rospy.is_shutdown():
            if self.state == State.LF:
                if DEBUG:
                    self.loginfo("<v:{} o:{}>".format(self.twist.v,self.twist.omega))

                if self.proportional is None:
                    self.twist.omega = 0
                    self.last_error = 0
                else:

                    P = - self.proportional * self.P

                    # D Term
                    d_error = (self.proportional - self.last_error) / (rospy.get_time() - self.last_time)
                    self.last_error = self.proportional
                    self.last_time = rospy.get_time()
                    D = d_error * self.D

                    if self.tof_det_range < 0.2:
                        self.twist.v = 0.0
                        self.twist.omega = 0.0
                        rospy.signal_shutdown("Finished Parking")

                    else:
                        self.twist.v = self.velocity
                        self.twist.omega = P + D
                        self.loginfo(self.twist.omega)

                if DEBUG:
                    self.loginfo([self.twist.v, self.twist.omega])

                self.vel_pub.publish(self.twist)
            else:
                if DEBUG:
                    self.loginfo("<v:{} o:{} slow>".format(self.twist.v, self.twist.omega))
                self.vel_pub.publish(self.twist)
                rospy.sleep(rospy.Duration(0.5))
                self.vel_pub.publish(Twist2DStamped(v=0, omega=0))
            self.rate.sleep()

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
