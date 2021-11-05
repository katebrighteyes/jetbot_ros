#!/usr/bin/env python

import cv2, rospy, time
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

#TODO 1 
bridge = CvBridge()

def img_callback(img_data):
  #TODO 2
  global bridge
  global cv_image
  cv_image = bridge.imgmsg_to_cv2(img_data, "bgr8")

#TODO 3
rospy.init_node('Camera_sub_node')
rospy.Subscriber("/usb_cam/image_raw/", Image, img_callback)

time.sleep(1.5)

while not rospy.is_shutdown():
  #TODO 4
  cv2.imshow('original', cv_image)

  if cv2.waitKey(1) & 0Xff == ord('q'):
    break

cv2.destroyAllWindows()
