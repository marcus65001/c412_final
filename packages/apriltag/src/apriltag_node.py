#!/usr/bin/env python3
import rospy
from duckietown.dtros import DTROS, NodeType, TopicType
from sensor_msgs.msg import CompressedImage, CameraInfo
from geometry_msgs.msg import Pose, Quaternion, Point, TransformStamped, Transform, Vector3, PoseStamped
from typing import cast
import numpy as np
import cv2
from turbojpeg import TurboJPEG, TJPF_GRAY
from cv_bridge import CvBridge
import yaml
from dt_apriltags import Detector
from duckietown_msgs.msg import AprilTagDetection
from std_msgs.msg import String, Int32, Bool
import tf
from tf2_ros import TransformBroadcaster, Buffer, TransformListener


class TagDetectorNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(TagDetectorNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self.veh = rospy.get_param("~veh")

        # parameters and internal objects
        self.image = None
        self.cam_info = None
        self._bridge = CvBridge()
        self._at_detector = Detector(families='tag36h11',
                                     nthreads=1,
                                     quad_decimate=1.0,
                                     quad_sigma=0.0,
                                     refine_edges=1,
                                     decode_sharpening=0.25,
                                     debug=0)
        self._at_detector_cam_para = None

        self.ci_cam_matrix = None
        self.ci_cam_dist = None

        # color mappings
        self.tag_cat_id = {
            "ua": [93, 94, 200, 201],
            "t": [58, 62, 133, 153],
            "stop": [162, 169],
            # "other":[227]
        }
        self.tag_loc = {
            200: (0.17, 0.17),
            201: (1.65, 0.17),
            94: (1.65, 2.84),
            93: (0.17, 2.84),
            153: (1.75, 1.252),
            133: (1.253, 1.755),
            58: (0.574, 1.259),
            62: (0.075, 1.755),
            169: (0.574, 1.755),
            162: (1.253, 1.253)
        }
        self.tag_all_id = np.concatenate([self.tag_cat_id[i] for i in self.tag_cat_id])
        self.tag_color = {
            None: "WHITE",
            "ua": "GREEN",
            "stop": "RED",
            "t": "BLUE",
            "other": "LIGHT_OFF"
        }
        self.tag_det = None
        self.tag_det_dist = 1.4

        # subscriber
        self.sub_comp_img = rospy.Subscriber('~cam', CompressedImage, self.cb_img)  # camera image topic
        self.sub_cam_info = rospy.Subscriber('~cam_info', CameraInfo, self.cb_cam_info)  # camera info topic
        self.sub_shutdown = rospy.Subscriber('~shutdown', Bool, self.cb_shutdown)  # shutdown topic

        # publisher
        self.pub = rospy.Publisher(
            "~image/compressed",
            CompressedImage,
            queue_size=1,
            dt_topic_type=TopicType.VISUALIZATION,
            dt_help="The stream of JPEG compressed images from the modified camera feed",
        )

        self.pub_at = rospy.Publisher(
            "~tag",
            AprilTagDetection,
            queue_size=1
        )


    def read_image(self, msg):
        try:
            img = self._bridge.compressed_imgmsg_to_cv2(msg)
            if (img is not None) and (self.image is None):
                self.log("got first msg")
            return img
        except Exception as e:
            self.log(e)
            return np.array([])

    def cb_cam_info(self, msg):
        if not self.cam_info:
            self.cam_info = msg
            self.log('read camera info')
            self.log(self.cam_info)
            # init camera info matrices
            self.ci_cam_matrix = np.array(self.cam_info.K).reshape((3, 3))
            self.ci_cam_dist = np.array(self.cam_info.D).reshape((1, 5))

            # init tag detector parameters
            camera_matrix = np.array(self.cam_info.K).reshape((3, 3))
            self._at_detector_cam_para = (
                camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2])

    def cb_shutdown(self, msg):
        if msg.data == True:
            self.loginfo("shutdown signal received")
            rospy.signal_shutdown("finish")

    def undistort(self, u_img):
        h, w = u_img.shape[:2]
        # h=self.cam_info.height
        # w=self.cam_info.width
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.ci_cam_matrix, self.ci_cam_dist, (w, h), 1, (w, h))
        dst = cv2.undistort(u_img, self.ci_cam_matrix, self.ci_cam_dist, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        # dst = cv2.UMat(dst,[y,y+h],[x,x+w])
        return dst

    def tag_id_to_color(self, id):
        cat = None
        for k, v in self.tag_cat_id.items():
            if id in v:
                cat = k
        return self.tag_color[cat] if cat else self.tag_color['other']

    def draw_segment(self, image, pt_A, pt_B, color):
        defined_colors = {
            'RED': ['rgb', [1, 0, 0]],
            'GREEN': ['rgb', [0, 1, 0]],
            'BLUE': ['rgb', [0, 0, 1]],
            'yellow': ['rgb', [1, 1, 0]],
            'purple': ['rgb', [1, 0, 1]],
            'cyan': ['rgb', [0, 1, 1]],
            'WHITE': ['rgb', [1, 1, 1]],
            'LIGHT_OFF': ['rgb', [0, 0, 0]]}
        _color_type, [r, g, b] = defined_colors[color]
        cv2.line(image, pt_A, pt_B, (b * 255, g * 255, r * 255), 5)
        return image

    def tag_detect(self, img):
        tags = self._at_detector.detect(img, True, self._at_detector_cam_para, 0.051)
        # print(tags)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        dist = np.inf
        rcand = None
        for r in tags:
            (ptA, ptB, ptC, ptD) = r.corners
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))

            # draw the bounding box
            color = self.tag_id_to_color(r.tag_id)
            self.draw_segment(img, ptA, ptB, color)
            self.draw_segment(img, ptB, ptC, color)
            self.draw_segment(img, ptC, ptD, color)
            self.draw_segment(img, ptD, ptA, color)

            # draw the center
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            cv2.circle(img, (cX, cY), 5, (0, 0, 255), -1)

            # dist
            if (tdist := np.linalg.norm(r.pose_t)) < dist:
                dist = tdist
                rcand = r
        # self.loginfo("tag: {}".format(rcand))
        if rcand:
            if dist > self.tag_det_dist:
                self.log("tag {} too far ({}), ignored".format(rcand.tag_id, dist))
                rcand = None
        self.tag_det = rcand

        return img

    def cb_img(self, msg):
        # image callback
        if self._bridge and (self.ci_cam_matrix is not None):
            # rectify
            # uimg=cv2.UMat(self.read_image(msg))
            self.image = self.read_image(msg)

    def _matrix_to_quaternion(self,r):
        T = np.array(((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1)), dtype=np.float64)
        T[0:3, 0:3] = r
        return tf.transformations.quaternion_from_matrix(T)

    def run(self):
        rate = rospy.Rate(2)
        while not rospy.is_shutdown():
            if self.image is not None:
                # publish image
                ud_image = self.undistort(self.image)
                # grayscale
                g_image = cv2.cvtColor(ud_image, cv2.COLOR_BGR2GRAY)
                # tag detection
                td_image = self.tag_detect(g_image)
                # tag id message
                if self.tag_det is not None:
                    q = self._matrix_to_quaternion(self.tag_det.pose_R)
                    p = self.tag_det.pose_t.T[0]
                    # create single tag detection object
                    tag_msg = AprilTagDetection(
                        transform=Transform(
                            translation=Vector3(x=p[0], y=p[1], z=p[2]),
                            rotation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]),
                        ),
                        tag_id=self.tag_det.tag_id,
                        tag_family=str(self.tag_det.tag_family),
                        hamming=self.tag_det.hamming,
                        decision_margin=self.tag_det.decision_margin,
                        homography=self.tag_det.homography.flatten().astype(np.float32).tolist(),
                        center=self.tag_det.center.tolist(),
                        corners=self.tag_det.corners.flatten().tolist(),
                        pose_error=self.tag_det.pose_err,
                    )
                    self.pub_at.publish(tag_msg)

                rate.sleep()


if __name__ == '__main__':
    # create the node
    node = TagDetectorNode(node_name='apriltag_node')
    # keep spinning
    node.run()
    rospy.spin()
