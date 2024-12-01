#!/usr/bin/env python3

import os
import rospy
import cv2
import numpy as np
import torch
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
from duckietown.dtros import DTROS, NodeType
from ultralytics import YOLO


class ObjectDetectionNode(DTROS):

    def __init__(self, node_name):
        
        # Initialize the DTROS parent class
        super(ObjectDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        
        self.model = None

        # Static parameters
        self._vehicle_name = os.environ['VEHICLE_NAME']
        # Subscribe to compressed image topic
        self._image_subscriber = rospy.Subscriber(
            f'/{self._vehicle_name}/camera_node/image/compressed',
            CompressedImage,
            self.image_callback,
            queue_size=1,
            buff_size=2**24
        )
        # Publisher for detected object coordinates
        self._coordinates_publisher = rospy.Publisher(
            f'/{self._vehicle_name}/object_coordinates',
            Point,
            queue_size=10
        )
        # Load YOLOv5 model
        self.model = self.load_yolo_model()

    def load_yolo_model(self):
        """Load the YOLOv5 model."""
        try:
            rospy.loginfo("Loading YOLOv5 model...")
            model =  YOLO('yolov5s.pt')
            # model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5s model
            model.eval()  # Set to evaluation mode
            rospy.loginfo("YOLOv5 model loaded successfully.")
            return model
        except Exception as e:
            rospy.logerr(f"Error loading YOLOv5 model: {e}")
            return None

    def image_callback(self, msg):
        """Callback function to process compressed images."""
        if not self.model:
            rospy.logerr("YOLOv5 model is not initialized. Skipping inference.")
            return

        try:
            # Decode compressed image
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                rospy.logerr("Failed to decode compressed image.")
                return

            # Convert BGR to RGB for YOLOv5
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform inference
            with torch.no_grad():  # Disable gradient calculation for inference
                results = self.model(frame_rgb)

            # Process detections
            detections = results.xyxy[0].cpu().numpy()  # Bounding boxes
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                label = self.model.names[int(cls)]
                if conf > 0.5:  # Confidence threshold
                    rospy.loginfo(f"Detected {label} with confidence {conf:.2f}")
                    # Calculate the center of the bounding box
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    # Publish coordinates as a Point message
                    point = Point(x=center_x, y=center_y, z=0)
                    self._coordinates_publisher.publish(point)

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")


if __name__ == '__main__':
    # Create the node
    node = ObjectDetectionNode(node_name='object_detection_node')
    # Keep the node running
    rospy.spin()

