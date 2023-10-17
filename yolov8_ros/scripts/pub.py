#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool

def publisher_node():
    # Initialize the ROS node
    rospy.init_node('obstacle_publisher', anonymous=True)

    # Create a publisher for the "is_obstacle" topic with String messages
    pub = rospy.Publisher('obstacle_scan', Bool, queue_size=10)

    # Rate at which to publish messages (in Hz)
    rate = rospy.Rate(1)  # 1 message per second

    while not rospy.is_shutdown():
        # Create a String message
        message = Bool()
        # message.data = False
        message.data = True

        # Publish the message
        pub.publish(message)

        # Sleep to control the publishing rate
        rate.sleep()

if __name__ == '__main__':
    try:
        publisher_node()
    except rospy.ROSInterruptException:
        pass
