#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge
import cv2

class CameraNode:
    def __init__(self):
        rospy.init_node('camera_node', anonymous=True)

        # Publishers for raw and compressed image topics
        self.raw_pub = rospy.Publisher('/hercules/camera_node/image/raw', Image, queue_size=10)
        self.compressed_pub = rospy.Publisher('/hercules/camera_node/image/compressed', CompressedImage, queue_size=10)
        self.info_pub = rospy.Publisher('/hercules/camera_node/camera_info', CameraInfo, queue_size=10)

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Simulating or reading from an actual camera device
        self.cap = cv2.VideoCapture(0)  # Adjust the device ID if necessary

        if not self.cap.isOpened():
            rospy.logerr("Failed to open the camera")
            rospy.signal_shutdown("Camera initialization failed")
        
        rospy.loginfo("Camera node started successfully")
    
    def publish_images(self):
        rate = rospy.Rate(10)  # Publish at 10 Hz
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                rospy.logerr("Failed to capture image")
                continue

            try:
                # Publish raw image
                raw_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                raw_msg.header.stamp = rospy.Time.now()
                raw_msg.header.frame_id = "camera_frame"
                self.raw_pub.publish(raw_msg)

                # Publish compressed image
                compressed_msg = CompressedImage()
                compressed_msg.header.stamp = rospy.Time.now()
                compressed_msg.header.frame_id = "camera_frame"
                compressed_msg.format = "jpeg"
                compressed_msg.data = cv2.imencode('.jpg', frame)[1].tobytes()
                self.compressed_pub.publish(compressed_msg)

                # Publish camera info (optional)
                camera_info = CameraInfo()
                camera_info.header.stamp = rospy.Time.now()
                camera_info.header.frame_id = "camera_frame"
                # Set intrinsic and distortion parameters as needed
                self.info_pub.publish(camera_info)

            except Exception as e:
                rospy.logerr(f"Error publishing images: {e}")
            
            rate.sleep()

    def shutdown(self):
        rospy.loginfo("Shutting down camera node")
        self.cap.release()

if __name__ == '__main__':
    try:
        node = CameraNode()
        node.publish_images()
    except rospy.ROSInterruptException:
        pass
    finally:
        node.shutdown()

