#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

class BoundingBoxWithCameraFeed:
    def __init__(self):
        rospy.init_node('bounding_box_with_camera_node', anonymous=True)

        # Publisher for the modified camera feed
        self.image_pub = rospy.Publisher('/camera_with_bounding_boxes', Image, queue_size=10)

        # Subscriber to the Duckiebot camera feed
        self.image_sub = rospy.Subscriber('/duckiebot/camera_node/image/raw', Image, self.image_callback)

        self.bridge = CvBridge()

    def image_callback(self, data):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")

            # Perform object detection and draw bounding boxes
            cv_image_with_boxes = self.draw_bounding_boxes(cv_image)

            # Publish the modified image
            image_msg = self.bridge.cv2_to_imgmsg(cv_image_with_boxes, encoding="bgr8")
            self.image_pub.publish(image_msg)

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def draw_bounding_boxes(self, image):
        """
        Draw bounding boxes on the image.
        Replace the dummy bounding boxes with your object detection results.
        """
        # Example bounding boxes
        detections = [
            {"x": 100, "y": 50, "w": 80, "h": 100, "label": "Object1"},
            {"x": 300, "y": 200, "w": 50, "h": 50, "label": "Object2"},
        ]

        for obj in detections:
            # Draw the bounding box
            x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Add the label
            label = obj["label"]
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image

    def run(self):
        rospy.loginfo("Bounding Box with Camera Feed Node Running")
        rospy.spin()

if __name__ == '__main__':
    try:
        node = BoundingBoxWithCameraFeed()
        node.run()
    except rospy.ROSInterruptException:
        pass

