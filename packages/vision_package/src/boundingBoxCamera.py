#!/usr/bin/env python
import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
from std_msgs.msg import String
from duckietown.dtros import DTROS, NodeType
import cv2
import numpy as np
import torch

class YOLOv5DetectorCompressed(DTROS):
    def __init__(self, nodeName):  
        super(yolov5_detector_compressed, self).__init__(node_name=nodeName, node_type=NodeType.GENERIC)
        self.model = None

        # Subscribe to the Duckiebot's compressed image topic
        self.image_sub = rospy.Subscriber('/hercules/camera_node/image/compressed', CompressedImage, self.image_callback)
       
        # Publisher for detected person's coordinates
        self.person_pub = rospy.Publisher('/hercules/person_coordinates', Point, queue_size=10)
       
        # Load YOLOv5 model
        try:
            rospy.loginfo("Loading YOLOv5 model...")
            self.model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Default YOLOv5s model
            self.model.eval()  # Added model evaluation mode
            rospy.loginfo("YOLOv5 model loaded successfully.")
        except Exception as e:
            rospy.logerr(f"Failed to load YOLOv5 model: {e}")
            self.model = None  # Fail gracefully
            self.loadModel()

    def loadModel(self):
        try:
            rospy.loginfo("Loading YOLOv5 model...")
            self.model = torch.hub.load("ultralytics/yolov5", "yolov5s")
            self.model.eval()  # Put the model in evaluation mode for inference
            rospy.loginfo("YOLOv5 model loaded successfully.")
        except Exception as e:
            rospy.logerr(f"Failed to load YOLOv5 model: {e}")
            self.model = None

    def image_callback(self, msg):
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
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Added color space conversion

            # Perform inference
            with torch.no_grad():  # Added for efficiency
                results = self.model(frame_rgb)

            # Extract and process detections
            detections = results.xyxy[0].cpu().numpy()  # Bounding boxes
            person_coordinates = None
            
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                label = self.model.names[int(cls)]
                if label == 'person' and conf > 0.5:
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    person_coordinates = Point(center_x, center_y, 0)
                    rospy.loginfo(f"Person detected at: x={center_x}, y={center_y}")
                    break

            # Publish person's coordinates
            if person_coordinates:
                self.person_pub.publish(person_coordinates)
                rospy.loginfo("Person coordinates published.")
            else:
                rospy.loginfo("No person detected.")

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

if __name__ == '__main__':
    try:
        detector = YOLOv5DetectorCompressed(nodeName='yolov5_detector_compressed')
        if detector.model is not None:  
            rospy.spin()
    except rospy.ROSInterruptException:
        pass
