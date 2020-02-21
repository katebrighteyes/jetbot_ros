#!/usr/bin/env python

import cv2, time
import rospy, rospkg
import numpy as np
from std_msgs.msg import String
from OpenCV_Utils import *

def processingSingleImage(image):
    result = imageCopy(image)
    result = convertColor(result, cv2.COLOR_BGR2GRAY)
    result = cannyEdge(result, 100, 200)
    height, width = result.shape[:2]
    src_pt1 = [int(width*0.35), int(height*0.65)]
    src_pt2 = [int(width*0.65), int(height*0.65)]
    src_pt3 = [width, height]
    src_pt4 = [0, height]
    dst_pt1 = [int(width*0.1),0]
    dst_pt2 = [int(width*0.9),0]
    dst_pt3 = [int(width*0.9),height]
    dst_pt4 = [int(width*0.1),height]
    src_pts = np.float32([src_pt1, src_pt2, src_pt3, src_pt4])
    dst_pts = np.float32([dst_pt1, dst_pt2, dst_pt3, dst_pt4])
    result = imagePerspectiveTransformation(result, src_pts, dst_pts)
    lines = houghLinesP(result, 1, np.pi/180, 40)

    empty = np.zeros((height, width), np.uint8)

    result_1 = drawHoughLinesP(empty, lines)
    result_2 = imagePerspectiveTransformation(result_1, dst_pts, src_pts)
    #result_3 = addImage(result_2, resultcolor)
    result = addImage(result_2, image)
    return result

cv_image = np.empty(shape=[0])

video_path = str(rospkg.RosPack().get_path('test_ros_cv')) + "/video/solidWhiteRight.mp4"
print(video_path)
cap = cv2.VideoCapture(video_path)

rospy.init_node('ros_cv_node')

pub = rospy.Publisher('test_cv', String) 
# topic name.

#rate = rospy.Rate(10) #2 times per second

while not rospy.is_shutdown():
    ret, cv_image = cap.read()

    result = processingSingleImage(cv_image)

    if cv2.waitKey(1) & 0Xff == 27:
        break

    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
        continue
   
    cv2.imshow("origin", result)

    pub.publish('Hello ROS')
    #time.sleep(0.01)

cv2.destroyAllWindows()
