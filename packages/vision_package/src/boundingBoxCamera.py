#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

class BoundingBoxWithCameraFeed:
    def __init__(self):
        rospy.init_node('boundingBoxCamera_node', anonymous=True)

        self.image_pub = rospy.Publisher('/camera_with_bounding_boxes', Image, queue_size=10)
        self.image_sub = rospy.Subscriber('/hercules/camera_node/image/compressed', Image, self.image_callback)

        self.bridge = CvBridge()

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
            cv_image_with_boxes = self.draw_bounding_boxes(cv_image)
            image_msg = self.bridge.cv2_to_imgmsg(cv_image_with_boxes, encoding="bgr8")
            self.image_pub.publish(image_msg)

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def draw_bounding_boxes(self, image):
        detections = [
            {"x": 50, "y": 50, "w": 100, "h": 100, "label": "Test Object"}
            #{"x": 100, "y": 50, "w": 80, "h": 100, "label": "Person", "confidence": 0.95},
        ]

        for obj in detections:
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

