#!/usr/bin/env python3

import rospy

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage, Range
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
STOP_MASK_L = [(0, 70, 50), (10, 255, 255)]
STOP_MASK_H = [(170, 70, 50), (180, 255, 255)]
DUCK_MASK=[(6,50,0),(26,255,255)]
STOP_MASK_BLUE=[(105,95,95),(120,255,255)]
DEBUG = True


class State(Enum):
    LF = auto()
    LF_ENGLISH = auto()
    PARK = auto()
    STRAIGHT=auto()
    LEFT= auto()


class ConstantControl:
    def __init__(self, value):
        self.value=value

    def get(self):
        return self.value

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
        if self.proportional is None:
            return None
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
        self.cap_omega=6.5
        self.cap_v=0.7

    def get_velocity(self,node):
        return 0

    def get_omega(self,node):
        return 0

    def get_twist(self,node):
        return Twist2DStamped(v=self.get_velocity(node), omega=self.get_omega(node))


class LF_Controller(Controller):
    def __init__(self,velocity=0.3,left_turn_omega=4):
        super().__init__()
        self.PD_omega = PD(-0.045, 0.0035)
        self.PD_all = PD(1.0,0.002)
        self.PD_all.set_disable(1.0)
        self.constant_v = ConstantControl(velocity)
        self.left_turn_omega=left_turn_omega

    def __repr__(self):
        return "<PD_omega={} PD_all={}>".format(self.PD_omega,self.PD_all)

    def get_omega(self,node):
        omega_cand=self.PD_omega.get()
        if omega_cand is None:
            if node.state==State.LEFT:
                omega_cand=self.left_turn_omega
            else:
                omega_cand=0
        return max(min(omega_cand,self.cap_omega)*self.PD_all.get(),-self.cap_omega)

    def get_velocity(self,node):
        return max(0,min(self.constant_v.get()*self.PD_all.get(),self.cap_v))


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

        self.tag_to_state = {
            56:State.STRAIGHT,
            50:State.LEFT
        }

        # Properties
        self.state = State.LF
        self.controller = LF_Controller(0.3)
        self.pause_stop_detection = False
        self.tag_det_id = 0
        self.tag_det=None
        self.det_duck = False
        self.offset_sign = 1
        self.lf_img=None

        # Timers
        self.stopping_timer = None
        self.masking_timer = None
        self.pause_timer=None
        self.avoid_timer=None

        # Publishers & Subscribers
        self.pub = rospy.Publisher("/" + self.veh + "/output/image/mask/compressed",
                                   CompressedImage,
                                   queue_size=1)

        self.vel_pub = rospy.Publisher("/" + self.veh + "/car_cmd_switch_node/cmd",
                                       Twist2DStamped,
                                       queue_size=1)

        self.sub = rospy.Subscriber("/" + self.veh + "/camera_node/image/compressed",
                                    CompressedImage,
                                    self.callback,
                                    queue_size=1,
                                    buff_size="20MB")
        self.tagid_sub = rospy.Subscriber("~tagid",
                                          Int32,
                                          self.cb_tagid,
                                          queue_size=1)
        self.tof_sub = rospy.Subscriber("~tof_range",
                                        Range,
                                        self.cb_tof,
                                        queue_size=1)


        # Shutdown hook
        rospy.on_shutdown(self.hook)

    # Calculates the midpoint of the contoured object
    def midpoint(self, x, y, w, h):
        mid_x = int(x + (((x + w) - x) / 2))
        mid_y = int(y + (((y + h) - y)))
        return (mid_x, mid_y)

    def cb_img_lf(self, img):
        if not isinstance(self.controller,LF_Controller):
            return
        # Part for Lane Following Detection
        crop = img[330:-1, :, :]
        if self.state in {State.LEFT,State.STRAIGHT}:
            crop[:,-200:,:]=0
        self.lf_img=crop
        if DEBUG:
            rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(crop))
            self.pub.publish(rect_img_msg)
        crop_height, crop_width, _ = crop.shape

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # lane follow
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
                self.controller.PD_omega.proportional = cx - int(crop_width / 2) + 220*self.offset_sign  # lf_offset
                if DEBUG:
                    cv2.drawContours(crop, contours, max_idx, (0, 255, 0), 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
            except:
                pass
        else:
            self.controller.PD_omega.proportional = None

        # stop line (red)
        if not self.pause_stop_detection and self.tag_det_id != 163:
            mask_stop_l=cv2.inRange(hsv, *STOP_MASK_L)
            mask_stop_h=cv2.inRange(hsv, *STOP_MASK_H)
            mask_stop = mask_stop_h + mask_stop_l

            cont_stop, hierarchy_stop = cv2.findContours(mask_stop,
                                                         cv2.RETR_EXTERNAL,
                                                         cv2.CHAIN_APPROX_NONE)
            if len(cont_stop) > 0:
                max_contour = max(cont_stop, key=cv2.contourArea)
                m_area = cv2.contourArea(max_contour)
                # self.loginfo("contour area: {}".format(m_area))
                if m_area > 3500:
                    M = cv2.moments(max_contour)
                    try:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        # enable PD controller to stop
                        self.controller.PD_all.proportional = 1.0 - cy / (crop_height + 20)  # lf_offset
                        if self.stopping_timer is None:
                            if DEBUG:
                                self.loginfo("Stopping")
                            self.controller.PD_all.set_disable(None)
                            self.stopping_timer = rospy.Timer(rospy.Duration(4.5), self.cb_stopping_timeup,
                                                              oneshot=True)
                            self.state = self.tag_to_state.get(self.tag_det_id, State.LF)
                            self.loginfo(self.state)
                            if self.state in {State.STRAIGHT, State.LEFT}:
                                self.loginfo("Go straight/left, mask right")
                                self.masking_timer = rospy.Timer(rospy.Duration(10.0), self.cb_masking_timeup,
                                                                 oneshot=True)
                        if DEBUG:
                            cv2.drawContours(crop, contours, max_idx, (255, 0, 0), 3)
                            cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
                    except:
                        pass
                # elif m_area>6000 and not self.avoid_timer:
                elif False:
                    # vehicle detection
                    if DEBUG:
                        self.loginfo("Large stop contour, avoid/LF_ENGLISH.")
                    self.state = State.LF_ENGLISH
                    self.offset_sign = -1
                    self.controller.PD_omega.reset()
                    self.avoid_timer = rospy.Timer(rospy.Duration(5.0), self.cb_avoiding_timeup, oneshot=True)
            elif self.stopping_timer is not None:
                # lost stop line detection
                self.controller.PD_all.proportional = 0.0

    def cb_tag(self,msg):
        if msg.data:
            self.tag_det=msg
            self.tag_det_id = self.tag_det.tag_id
            stop_start_distance=0.5
            min_distance=0.2
            dist=self.tag_det.transform.translation.z
            if self.tag_det_id == 163 and dist < stop_start_distance:
                # enable PD controller to stop
                if self.stopping_timer is None:
                    if DEBUG:
                        self.loginfo("Stopping (blue)")
                    self.controller.PD_all.set_disable(None)
                    self.stopping_timer = rospy.Timer(rospy.Duration(4.5), self.cb_stopping_timeup,
                                                      oneshot=True)
                    self.state = self.tag_to_state.get(self.tag_det_id, State.LF)
                    self.loginfo(self.state)
                else:
                    # update error
                    self.controller.PD_all.proportional = max(0,dist-0.2)/(stop_start_distance-min_distance)

    def cb_tof(self,msg):  # tof sensor, Range msg, in meters
        self.tof_det_range = msg.range if msg.min_range>msg.range>msg.max_range else np.inf


    def callback(self, msg):
        img = self.jpeg.decode(msg.data)
        if self.state!=State.PARK:
            self.cb_img_lf(img)
        else:
            # parking
            pass

    def cb_stopping_timeup(self,te):
        if self.det_duck:
            if DEBUG:
                self.loginfo("Stopping time up, but think of the ducklings!")
            self.stopping_timer = rospy.Timer(rospy.Duration(3.0), self.cb_stopping_timeup, oneshot=True)
            return
        # try to see duck

        if DEBUG:
            self.loginfo("Stopping time up. Pause stop detection.")
        self.pause_stop_detection = True
        if self.state!=State.STRAIGHT:
            self.loginfo("Short pause.")
            self.pause_timer = rospy.Timer(rospy.Duration(3.0), self.cb_pause_timeup, oneshot=True)
        else:
            self.loginfo("Long pause.")
            self.pause_timer = rospy.Timer(rospy.Duration(9.0), self.cb_pause_timeup, oneshot=True)
        self.controller.PD_all.set_disable(1.0)
        self.stopping_timer=None
        return

    def cb_pause_timeup(self,te):
        if DEBUG:
            self.loginfo("Pause time up.")
        self.pause_timer=None
        self.pause_stop_detection=False
        return

    def cb_masking_timeup(self,te):
        if DEBUG:
            self.loginfo("Masking time up.")
        self.masking_timer=None
        self.state=State.LF
        return

    def cb_avoiding_timeup(self,te):
        if DEBUG:
            self.loginfo("Avoiding time up, back to LF.")
        self.state = State.LF
        self.offset_sign = 1
        self.controller.PD_omega.reset()
        self.avoid_timer = None
        return

    def cb_duck_det(self,te):
        #duck detection
        template = cv2.imread('/data/duck.png')
        res = cv2.matchTemplate(self.lf_img, template, cv2.TM_CCORR_NORMED)
        threshold = 0.9
        loc = np.where(res >= threshold)
        if len(loc[0])>100:
            if DEBUG:
                self.loginfo("See duck")
            self.det_duck=True
        else:
            self.det_duck=False


    def drive(self):
        self.loginfo(self.controller)
        self.vel_pub.publish(self.controller.get_twist(self))

    def hook(self):
        self.loginfo("SHUTTING DOWN")
        twist=Twist2DStamped(v=0, omega=0)
        self.vel_pub.publish(twist)
        for i in range(8):
            self.vel_pub.publish(twist)


if __name__ == "__main__":
    node = ControlNode("control_node")
    rate = rospy.Rate(8)  # 8hz
    rospy.sleep(rospy.Duration(5))
    # t_duck_det=rospy.Timer(rospy.Duration(0.5),node.cb_duck_det)
    while not rospy.is_shutdown():
        node.drive()
        rate.sleep()
