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

class ParkingNode(DTROS):

    def __init__(self, node_name):
        super(ParkingNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.node_name = node_name
        self.veh = rospy.get_param("~veh")
        self.stall_number = rospy.get_param("/stall",4)
        
        self.jpeg = TurboJPEG()

        self.loginfo("Initialized")

        # self.stall_numbers = {
        #     1:[207, 0.2, 90, 228],
        #     2:[226, 0.42, 90, 75],
        #     3:[228, 0.2, -90, 207],
        #     4:[75, 0.42, -90, 226],
        # }

        # self.stall_numbers = {
        #     1: [207, 0.62, -90],
        #     2: [226, 0.78, -90],
        #     3: [228, 0.62, 90],
        #     4: [75, 0.78, 90],
        # }

        self.stall_numbers = {
            1: [207, "Left", rospy.get_param("/o1",1.6)],
            2: [226, "Left", rospy.get_param("/o2",3.6)],
            3: [228, "Right", rospy.get_param("/o3",-1.9)],
            4: [75, "Right", rospy.get_param("/o4",-3.6)],
        }

        self.proportional = None
        self.proportional_stop = 0.0
        if ENGLISH:
            self.offset = -180
        else:
            self.offset = 180
        self.velocity = 0.3
        self.twist = Twist2DStamped(v=self.velocity, omega=0)

        self.P = 0.039
        self.D = -0.0025
        self.last_error = 0
        self.last_time = rospy.get_time()
        self.image_area = 640*480
        self.april_to_image_ratio = 0.0

        self.apriltag_corners = None
        self.at_area = 0.0
        self.apriltag_center = None
        # self.apriltag_id = None
        self.add_patch = True
        self.tof_det_range = None

        # self.enter = False
        self.timer = None
        self.stop_detection = True
        # self.remote = True

        self.v = 0.0
        self.omega = 0.0
        self.velocity = 0.3
        self.parking_velo = 0
        self.twist = Twist2DStamped(v=self.velocity, omega=0)

        # Publishers & Subscribers
        # self.sub_apriltag_info = rospy.Subscriber("~tag",
        #                                           AprilTagDetection,
        #                                           self.cb_apriltag_info,
        #                                           queue_size=1)
        self.sub_det = rospy.Subscriber("/" + self.veh + "/camera_node/image/compressed",
                                    CompressedImage,
                                    self.callback,
                                    queue_size=1,
                                    buff_size="20MB")

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

    # def cb_apriltag_info(self, msg):
    #
    #     if msg.tag_id:
    #         # Update detection info:
    #         self.apriltag_id = msg.tag_id
    #
    #         if msg.tag_id != 227:
    #             self.apriltag_center = msg.center
    #             self.apriltag_corners = msg.corners
    #             # self.loginfo("Corners = " + str(self.apriltag_corners))
    #             self.at_area = (self.apriltag_corners[4]-self.apriltag_corners[0])*(self.apriltag_corners[1]-self.apriltag_corners[5])
    #             self.april_to_image_ratio = self.at_area / self.image_area
                # self.loginfo("Area = " + str(self.at_area))

            # if self.stall_numbers.get(self.stall_number)[0] == self.apriltag_id:
            #     self.pixel_dist = math.sqrt(math.pow((self.midpoint_x - self.apriltag_center[0]), 2))
            #     self.prop_turn = self.pixel_dist

    def cb_tof(self, msg):  # tof sensor, Range msg, in meters

        self.tof_det_range = msg.range if msg.min_range < msg.range < msg.max_range else np.inf

    def callback(self, msg):

        if self.timer is None:
            self.stop_detection = True
            self.loginfo("Timer set")
            self.timer = rospy.Timer(rospy.Duration(5), self.cb_timer, oneshot=True)

        if self.stop_detection:
            return

        img = self.jpeg.decode(msg.data)
        if self.add_patch:
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

        if DEBUG:
            rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(crop))
            self.pub.publish(rect_img_msg)

    def parking_action(self):

        if self.stall_numbers.get(self.stall_number)[1] == 'Left':
            self.loginfo("Turning Left")
            self.twist.v = self.velocity
            self.twist.omega = self.stall_numbers.get(self.stall_number)[2]
        elif self.stall_numbers.get(self.stall_number)[1] == 'Right':
            self.loginfo("Turning Right")
            self.twist.v = self.velocity
            self.twist.omega = self.stall_numbers.get(self.stall_number)[2]
        else:
            self.twist.omega = 0
            self.twist.v = self.velocity
            self.last_error = 0

    def approach_stall(self):

        # Executes an action until the lane is detected again
        if self.proportional is None:
            # self.twist.omega = 0
            # self.last_error = 0
            self.parking_action()

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

        if DEBUG_TEXT:
            self.loginfo("TAG SELECTED: " + str(self.stall_number))
            self.loginfo("PROPORTIONAL: " + str(self.proportional))
            self.loginfo("TOF DISTANCE: " + str(self.tof_det_range))
            self.loginfo([self.twist.v, self.twist.omega])

        self.vel_pub.publish(self.twist)

    # def aproach_stall(self):
    #
    #     # self.loginfo(self.det_distance)
    #
    #     if self.apriltag_id == 227 and self.remote:
    #
    #         if self.det_distance > self.stall_numbers.get(self.stall_number)[1]:
    #             self.twist.v = self.velocity
    #             self.twist.omega = 0.0
    #         else:
    #             self.twist.v = 0.0
    #             rospy.sleep(2)
    #             self.P_controller()
    #             # if self.twist.v == 0.0:
    #             #     self.turn_into_stall()
    #             #     self.loginfo("Timer set")
    #             #     self.timer = rospy.Timer(rospy.Duration(0.15), self.cb_timer, oneshot= True)
    #     self.vel_pub.publish(self.twist)

    # def P_controller (self):
    #     # Executes an action until the lane is detected again
    #     self.loginfo(self.apriltag_id)
    #     if self.prop_turn is None:
    #         # self.twist.omega = 0
    #         # self.last_error = 0
    #         self.turn_into_stall()
    #         self.loginfo("Timer set")
    #         self.timer = rospy.Timer(rospy.Duration(0.15), self.cb_timer, oneshot=True)
    #         self.loginfo("Apriltag #" + str(self.apriltag_id))
    #     elif self.stall_numbers.get(self.stall_number)[0] == self.apriltag_id:
    #         # P Term
    #         p = - self.prop_turn * self.P_turn
    #         self.loginfo("P value" + str(p))
    #         # D Term
    #         # d_error = (self.proportional - self.last_error) / (rospy.get_time() - self.last_time)
    #         # self.last_error = self.proportional
    #         # self.last_time = rospy.get_time()
    #         # D = d_error * self.D
    #
    #         if self.det_distance < 0.5:
    #             self.twist.v = 0
    #             self.twist.omega = 0
    #         else:
    #             self.twist.v = self.velocity
    #             self.twist.omega = p
    #             self.loginfo("Omega value" + str(self.twist.omega))
    #
    # def turn_into_stall(self):
    #
    #     if self.stall_numbers.get(self.stall_number)[2] == 90:
    #         self.twist.omega = - 7
    #     else:
    #         self.twist.omega = 7
            
    def cb_timer(self, te):
        self.loginfo("Timer Up")
        self.stop_detection = False


    def drive(self):
        self.approach_stall()

    def hook(self):
        print("SHUTTING DOWN")
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)
        for i in range(8):
            self.vel_pub.publish(self.twist)


if __name__ == "__main__":
    node = ParkingNode("parking_node")
    rate = rospy.Rate(8)  # 8hz
    while not rospy.is_shutdown():
        node.drive()
        rate.sleep()
