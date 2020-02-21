#!/usr/bin/env python

import cv2, rospy, time
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from OpenCV_Utils import *

bridge = CvBridge()

def img_callback(img_data):
  global bridge
  global cv_image
  cv_image = bridge.imgmsg_to_cv2(img_data, "bgr8")

def CutRectROI(image, x1, y1, x2, y2):
    return image[y1:y2, x1:x2]

def processingSingleImage(image):
    result = imageCopy(image)
    height, width = image.shape[:2]
    roi_x1 = 0
    roi_y1 = int(height*0.7)
    roi_x2 = width
    roi_y2 = int(height*0.9)
    result = CutRectROI(image, roi_x1, roi_y1, roi_x2, roi_y2)

    #src_pt1 = [int(width*0.35), int(height*0.65)]
    #src_pt2 = [int(width*0.65), int(height*0.65)]
    #src_pt3 = [width, height]
    #src_pt4 = [0, height]

    #result = imageCopy(image)
    #result = cannyEdge(result, 100, 200)

    #dst_pt1 = [int(width*0.1),0]
    #dst_pt2 = [int(width*0.9),0]
    #dst_pt3 = [int(width*0.9),height]
    #dst_pt4 = [int(width*0.1),height]
    #src_pts = np.float32([src_pt1, src_pt2, src_pt3, src_pt4])
    #dst_pts = np.float32([dst_pt1, dst_pt2, dst_pt3, dst_pt4])
    #result = imagePerspectiveTransformation(result, src_pts, dst_pts)
    #lines = houghLinesP(result, 1, np.pi/180, 40)

    #empty = np.zeros((height, width), np.uint8)

    #result_1 = drawHoughLinesP(empty, lines)
    #result_2 = imagePerspectiveTransformation(result_1, dst_pts, src_pts)
    #result_3 = addImage(result_2, resultcolor)
    #result = addImage(result_2, image)
    return result


rospy.init_node('ros_cv_cam')
#rospy.Subscriber("/usb_cam/image_raw/", Image, img_callback)
rospy.Subscriber("/jetbot_camera/raw", Image, img_callback)

time.sleep(1.5)

while not rospy.is_shutdown():
  result = processingSingleImage(cv_image)
  
  cv2.imshow('original', result)

  if cv2.waitKey(1) & 0Xff == 27:
    break

cv2.destroyAllWindows()

