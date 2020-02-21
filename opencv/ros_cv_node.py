#!/usr/bin/env python

import cv2, time
import rospy, rospkg
import numpy as np
from std_msgs.msg import String


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

    if cv2.waitKey(1) & 0Xff == 27:
        break

    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
        continue
   
    cv2.imshow("origin", cv_image)

    pub.publish('Hello ROS')
    #time.sleep(0.01)

cv2.destroyAllWindows()
