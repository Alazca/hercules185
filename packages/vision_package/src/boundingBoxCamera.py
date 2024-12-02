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
        
        # Publisher for visualization (create publishers before subscriber)
        self.debug_publisher = rospy.Publisher(
            '/hercules/detection_visual/compressed',
            CompressedImage,
            queue_size=1
        )
        
        # Publisher for detected object coordinates
        self._coordinates_publisher = rospy.Publisher(
            'hercules/object_coordinates',
            Point,
            queue_size=10
        )
        
        # Load YOLOv5 model before creating subscriber
        self.model = self.load_yolo_model()
        
        # Subscribe to compressed image topic (create last)
        self._image_subscriber = rospy.Subscriber(
            'hercules/camera_node/image/compressed',
            CompressedImage,
            self.image_callback,
            queue_size=1,
            buff_size=2**24
        )
         

        # Load YOLOv5 model
        self.model = self.load_yolo_model()

    def load_yolo_model(self):
        """Load the YOLOv5 model."""
        try:
            rospy.loginfo("Loading YOLOv5 model...")
            self.model = YOLO('yolov5s.pt') 
            self.model.eval()  # Set to evaluation mode
            rospy.loginfo("YOLOv5 model loaded successfully.")
            return self.model
        except Exception as e:
            rospy.logerr(f"Error loading YOLOv5 model: {e}")
            self.model = None

    def publish_image(self, frame):
        try:
            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format  = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', frame)[1]).tobytes()
            self.debug_publisher.publish(msg)

        except Exception as e:
            rospy.logerr(f"Error Publishing image: {e}")
            
    def image_callback(self, msg):
        if self.model is None:
            rospy.logwarn_throttle(1, "YOLOv5 model not initialized yet. Skipping callback.")
            return
            
        try:
            # Decode compressed image
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                rospy.logerr("Failed to decode compressed image.")
                return

            # Keep a copy of original frame for visualization
            frame_viz = frame.copy()

            # Perform inference
            results = self.model(frame)
            
            # Process detections using the correct format
            for result in results:
                for box in result.boxes:
                    # Get confidence
                    conf = float(box.conf)
                    
                    if conf > 0.5:  # Confidence threshold
                        # Get class
                        cls = int(box.cls)
                        label = self.moodel.names[cls]
                        
                        # Get coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        
                        # Calculate center
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        # Publish coordinates
                        point = Point(x=center_x, y=center_y, z=conf)
                        self._coordinates_publisher.publish(point)
                        
                        # Draw on visualization frame
                        cv2.rectangle(frame_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame_viz, 
                                  f"{label} {conf:.2f}", 
                                  (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, 
                                  (0, 255, 0), 
                                  2)
            
            # Publish the visualization
            self.publish_image(frame_viz)
            
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())

if __name__ == '__main__':
    # Create the node
    node = ObjectDetectionNode(node_name='object_detection_node')
    # Keep the node running
    rospy.spin()

