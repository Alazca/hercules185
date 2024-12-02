#!/usr/bin/env python3

import os
import rospy
from geometry_msgs.msg import Point
from duckietown_msgs.msg import Twist2DStamped
from duckietown.dtros import DTROS, NodeType


class PersonFollowerNode(DTROS):
    def __init__(self, node_name):
        super(PersonFollowerNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC
        )

        movement_topic = "hercules/car_cmd_switch_node/cmd"
        coordinates_topic = "hercules/object_coordinates"

        # Publisher
        self._movement_publisher = rospy.Publisher(
            movement_topic, Twist2DStamped, queue_size=1
        )

        # Subscriber
        self._coordinates_subscriber = rospy.Subscriber(
            coordinates_topic, Point, self.process_coordinates, queue_size=1
        )

        # Parameters for following behavior
        self.safe_distance = 100  # Safe distance from the person (in pixels)
        self.image_center_x = 320  # Assuming 640x480 resolution
        self.distance_tolerance = 20  # Tolerance for safe distance
        self.offset_tolerance = 30  # Tolerance for lateral alignment

    def process_coordinates(self, msg):
        """Processes coordinates of detected object."""
        x, y, conf = msg.x, msg.y, msg.z

        if conf < 0.5:  # Ignore low-confidence detections
            return

        # Calculate movement commands
        twist = Twist2DStamped()
        distance_error = self.safe_distance - y
        offset_error = x - self.image_center_x

        # Adjust forward speed based on distance
        if abs(distance_error) > self.distance_tolerance:
            twist.v = 0.2 if distance_error > 0 else -0.2
        else:
            twist.v = 0  # Stop if within the safe distance

        # Adjust angular velocity based on lateral offset
        if abs(offset_error) > self.offset_tolerance:
            twist.omega = -0.005 * offset_error
        else:
            twist.omega = 0  # No turning if well-aligned

        self._movement_publisher.publish(twist)


if __name__ == "__main__":
    node = PersonFollowerNode(node_name="person_follower_node")
    rospy.spin()
