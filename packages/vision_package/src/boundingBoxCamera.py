#!/usr/bin/env python

import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch

class YOLOv5DetectorCompressed:
    def __init__(self):
        rospy.init_node('yolov5_detector_compressed', anonymous=True)

        # Subscribe to the Duckiebot's compressed image topic
        self.image_sub = rospy.Subscriber('/hercules/camera_node/image/compressed', CompressedImage, self.image_callback)

        # Publisher for person coordinates
        self.person_pub = rospy.Publisher('/person_coordinates', Point, queue_size=10)

        # Load YOLOv5 model
        try:
            rospy.loginfo("Loading YOLOv5 model...")
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='/code/assets/yolov5s.pt', force_reload=True)
            rospy.loginfo("YOLOv5 model loaded successfully.")
        except Exception as e:
            rospy.logerr(f"Failed to load YOLOv5 model: {e}")
            self.model = None  # Fail gracefully    

    def image_callback(self, msg):
        try:
            # Decode compressed image
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Run YOLOv5 inference
            results = self.model(frame)
            detections = results.xyxy[0].cpu().numpy()  # Bounding boxes

            person_coordinates = None

            # Process detections
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                label = self.model.names[int(cls)]

                if label == 'person' and conf > 0.5:
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    person_coordinates = Point(center_x, center_y, 0)
                    break

            if person_coordinates:
                self.person_pub.publish(person_coordinates)

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

if __name__ == '__main__':
    try:
        YOLOv5DetectorCompressed()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass



