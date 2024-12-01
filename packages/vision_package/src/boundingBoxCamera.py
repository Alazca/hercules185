# #!/usr/bin/env python
#
# import rospy
# from sensor_msgs.msg import CompressedImage, Image
# from cv_bridge import CvBridge, CvBridgeError
# import cv2
# import numpy as np
#
# class BoundingBoxWithCameraFeed:
#     def __init__(self):
#         rospy.init_node('boundingBoxCamera_node', anonymous=True)
#
#         self.image_sub = rospy.Subscriber('/hercules/camera_node/image/compressed', CompressedImage, self.image_callback)
#
#         self.image_pub = rospy.Publisher('/camera_with_bounding_boxes', Image, queue_size=10)
#
#         self.bridge = CvBridge()
#
#     def image_callback(self, data):
#         try:
#             rospy.loginfo("Compressed image received")
#             
#             # Decode the compressed image to OpenCV format
#             np_arr = np.frombuffer(data.data, np.uint8)
#             cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#             rospy.loginfo("Compressed image decoded to OpenCV format")
#
#             # Add bounding boxes
#             cv_image_with_boxes = self.draw_bounding_boxes(cv_image)
#
#             # Publish the processed image
#             image_msg = self.bridge.cv2_to_imgmsg(cv_image_with_boxes, encoding="bgr8")
#             self.image_pub.publish(image_msg)
#             rospy.loginfo("Published image with bounding boxes to /camera_with_bounding_boxes")
#         
#         except CvBridgeError as e:
#             rospy.logerr(f"CvBridge Error: {e}")
#         
#         except Exception as e:
#             rospy.logerr(f"Unexpected error: {e}")
#
#     def draw_bounding_boxes(self, image):
#         # Example bounding boxes
#         detections = [
#             {"x": 50, "y": 50, "w": 100, "h": 100, "label": "Test Object"}
#         ]
#
#         for obj in detections:
#             x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
#             # Draw rectangle
#             cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             # Add label
#             label = obj["label"]
#             cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#         return image
#
#     def run(self):
#         rospy.loginfo("Bounding Box with Camera Feed Node Running")
#         rospy.spin()
#
# if __name__ == '__main__':
#     try:
#         node = BoundingBoxWithCameraFeed()
#         node.run()
#     except rospy.ROSInterruptException:
#         pass
#
#             #{"x": 100, "y": 50, "w": 80, "h": 100, "label": "Person", "confidence": 0.95},

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
        
        # Subscribe to compressed image topic
        self.image_sub = rospy.Subscriber('/camera/image_raw/compressed', CompressedImage, self.image_callback)
        self.person_pub = rospy.Publisher('/person_coordinates', Point, queue_size=10)

        # Initialize CVBridge
        self.bridge = CvBridge()

        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='~/hercules185/assets/yolov5s.pt')

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



