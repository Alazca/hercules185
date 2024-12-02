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
        super(ObjectDetectionNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC
        )

        # Check CUDA availability and set up device
        cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda_available else "cpu")
        if cuda_available:
            # Get Jetson GPU info
            gpu_name = torch.cuda.get_device_name(0)
            rospy.loginfo(f"Using CUDA device: {gpu_name}")
            # Set memory configuration for Jetson
            torch.cuda.empty_cache()
            torch.cuda.memory.set_per_process_memory_fraction(
                0.8
            )  # Use 80% of available GPU memory
        else:
            rospy.logwarn("CUDA not available, using CPU")

        self.model = None
        self.process_every_n_frames = 2  # Adjust based on performance
        self.frame_count = 0
        self.image_size = 416  # Good balance for Jetson Nano

        self.debug_publisher = rospy.Publisher(
            "/hercules/detection_visual/compressed", CompressedImage, queue_size=1
        )

        self._coordinates_publisher = rospy.Publisher(
            "hercules/object_coordinates", Point, queue_size=10
        )

        # Load model after setting up CUDA
        self.model = self.load_yolo_model()

        self._image_subscriber = rospy.Subscriber(
            "hercules/camera_node/image/compressed",
            CompressedImage,
            self.image_callback,
            queue_size=1,
            buff_size=2**24,
        )

    def load_yolo_model(self):
        try:
            rospy.loginfo("Loading YOLOv5 model...")

            # Use YOLOv5n for better performance on Jetson Nano
            model = YOLO("yolov5s.pt")
            model.to(self.device)

            if self.device.type == "cuda":
                model.model.half()  # FP16 for faster inference
                rospy.loginfo("Model converted to FP16")

            # Warmup the model
            dummy_input = torch.zeros((1, 3, self.image_size, self.image_size)).to(
                self.device
            )
            if self.device.type == "cuda":
                dummy_input = dummy_input.half()

            rospy.loginfo("Warming up model...")
            with torch.no_grad():
                for _ in range(2):  # Reduced warmup iterations for faster startup
                    model(dummy_input)

            rospy.loginfo("Model loaded and ready")
            return model

        except Exception as e:
            rospy.logerr(f"Error loading YOLOv5 model: {e}")
            return None

    def preprocess_image(self, frame):
        """Optimize image preprocessing for Jetson"""
        # Use CUDA-optimized resize if available
        if hasattr(cv2, "cuda") and hasattr(cv2.cuda, "resize"):
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            gpu_frame = cv2.cuda.resize(gpu_frame, (self.image_size, self.image_size))
            return gpu_frame.download()
        else:
            return cv2.resize(frame, (self.image_size, self.image_size))

    def publish_image(self, frame):
        try:
            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            msg.data = np.array(cv2.imencode(".jpg", frame, encode_param)[1]).tobytes()
            self.debug_publisher.publish(msg)
        except Exception as e:
            rospy.logerr(f"Error publishing image: {e}")

    def image_callback(self, msg):
        if self.model is None:
            return

        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames != 0:
            return

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                return

            processed_frame = self.preprocess_image(frame)

            with (
                torch.cuda.amp.autocast()
                if self.device.type == "cuda"
                else torch.no_grad()
            ):
                results = self.model(
                    processed_frame, verbose=False, conf=0.5, iou=0.45, max_det=10
                )

            frame_viz = frame.copy()
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    conf = float(box.conf)
                    cls = int(box.cls)
                    label = self.model.names[cls]

                    # Scale coordinates back to original image size
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    scale_x = frame.shape[1] / self.image_size
                    scale_y = frame.shape[0] / self.image_size
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    self._coordinates_publisher.publish(
                        Point(x=center_x, y=center_y, z=conf)
                    )

                    cv2.rectangle(frame_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame_viz,
                        f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            self.publish_image(frame_viz)

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")


if __name__ == "__main__":
    node = ObjectDetectionNode(node_name="object_detection_node")
    rospy.spin()
