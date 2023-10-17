#!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan

def scan_callback(data):
    intensities_length = len(data.intensities)
    rospy.loginfo("Number of intensities: %d", intensities_length)

def subscriber_node():
    rospy.init_node('scan_subscriber', anonymous=True)

    rospy.Subscriber('/scan', LaserScan, scan_callback)

    rospy.spin()

if __name__ == '__main__':
    try:
        subscriber_node()
    except rospy.ROSInterruptException:
        pass
