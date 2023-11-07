#!/usr/bin/env python3

import cv2
import rospy
import numpy as np
from ultralytics import YOLO

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from yolov8_ros_msgs.msg import BoundingBox, BoundingBoxes

from std_msgs.msg import Bool

# STOP_OBJECT_LIST = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
STOP_OBJECT_LIST = ['person', 'skateboard']

class Yolo_Dect:
    def __init__(self):
        # load parameters
        weight_path = rospy.get_param('~weight_path', '')
        image_topic = rospy.get_param(
            '~image_topic', '/camera/color/image_raw')
        self.camera_frame = rospy.get_param('~camera_frame', '')
        conf = rospy.get_param('~conf', '0.5')
        self.visualize = rospy.get_param('~visualize', 'True')

        if (rospy.get_param('/use_cpu', 'false')):
            self.device = 'cpu'
        else:
            self.device = 'cuda'

        self.model = YOLO(weight_path)
        self.model.fuse()

        self.model.conf = conf
        self.color_image = Image()
        self.getImageStatus = True

        # Load class color
        self.classes_colors = {}

        self.is_obstacle = True

        self.color_sub = rospy.Subscriber(image_topic, Image, self.image_callback, queue_size=1, buff_size=52428800)


        # if no image messages
        while (not self.getImageStatus):
            rospy.loginfo("waiting for image.")
            rospy.sleep(2)


    def image_callback(self, image):
        self.getImageStatus = True
        if self.is_obstacle:
            self.boundingBoxes = BoundingBoxes()
            self.boundingBoxes.header = image.header
            self.boundingBoxes.image_header = image.header
            self.color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(
                image.height, image.width, -1)

            self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)

            modified_image = np.copy(self.color_image)
            blackout_top = 200
            blackout_bot = 80
            height = self.color_image.shape[0]
            for height_idx in range(height):
                blackout_width = int(blackout_top - ((blackout_top - blackout_bot)*height_idx/height))
                modified_image[height_idx, :blackout_width, :] = [121, 131, 248]
                modified_image[height_idx, -blackout_width:, :] = [121, 131, 248]

            results = self.model(modified_image, show=False, conf=self.model.conf, verbose=False)

            self.dectshow(results, image.height, image.width)

            cv2.waitKey(3)


    def obstacle_callback(self, data):
        if data.data == True:
            self.is_obstacle = True
        else:
            self.is_obstacle = False


    def dectshow(self, results, height, width):
        self.frame = results[0].plot()
        # print(str(results[0].speed['inference']))
        fps = 1000.0/ results[0].speed['inference']
        cv2.putText(self.frame, f'FPS: {int(fps)}', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)


        if self.visualize :
            cv2.imshow('YOLOv8_test', self.frame)


def main():
    rospy.init_node('yolov8_ros_test', anonymous=True)
    yolo_dect = Yolo_Dect()
    rospy.spin()


if __name__ == "__main__":
    main()