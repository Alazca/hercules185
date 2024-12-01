
#!/usr/bin/env python

import rospy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class BoundingBoxWithCameraFeed:
    def __init__(self):
        rospy.init_node('boundingBoxCamera_node', anonymous=True)

        self.image_sub = rospy.Subscriber('/hercules/camera_node/image/compressed', CompressedImage, self.image_callback)

        self.image_pub = rospy.Publisher('/camera_with_bounding_boxes', Image, queue_size=10)

        self.bridge = CvBridge()

    def image_callback(self, data):
        try:
            rospy.loginfo("Compressed image received")
            
            # Decode the compressed image to OpenCV format
            np_arr = np.frombuffer(data.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            rospy.loginfo("Compressed image decoded to OpenCV format")

            # Add bounding boxes
            cv_image_with_boxes = self.draw_bounding_boxes(cv_image)

            # Publish the processed image
            image_msg = self.bridge.cv2_to_imgmsg(cv_image_with_boxes, encoding="bgr8")
            self.image_pub.publish(image_msg)
            rospy.loginfo("Published image with bounding boxes to /camera_with_bounding_boxes")
        
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
        
        except Exception as e:
            rospy.logerr(f"Unexpected error: {e}")

    def draw_bounding_boxes(self, image):
        # Example bounding boxes
        detections = [
            {"x": 50, "y": 50, "w": 100, "h": 100, "label": "Test Object"}
        ]

        for obj in detections:
            x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
            # Draw rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Add label
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

            #{"x": 100, "y": 50, "w": 80, "h": 100, "label": "Person", "confidence": 0.95},
