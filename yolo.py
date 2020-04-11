# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.layers import Input
from yolo3.model import yolo_eval, tiny_yolo_body, magic_body_2
from yolo3.utils import letterbox_image
import os
import cv2


class YOLO:
    def __init__(self, model_path, anchros, num_classes, score_th, iou_th, model_input_shape):
        self.sess = K.get_session()
        self.model_path = model_path
        self.anchors = anchros
        self.num_classes = num_classes
        self.score_th = score_th
        self.iou_th = iou_th
        self.model_input_shape = model_input_shape
        self.num_outputs = 1
        self.boxes, self.scores, self.classes = self.generate()

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct msodel and load weights.
        num_anchors = len(self.anchors)

        input_shape = (self.model_input_shape[0], self.model_input_shape[1], 1)
        print('Model Info', model_path, num_anchors, self.num_classes, input_shape)
        self.yolo_model = magic_body_2(Input(shape=input_shape), num_anchors//self.num_outputs, self.num_classes)
        self.yolo_model.load_weights(self.model_path)

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))

        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                self.num_classes, self.input_image_shape, score_threshold=self.score_th, iou_threshold=self.iou_th)

        return boxes, scores, classes

    def close_session(self):
        self.sess.close()

    def detect_image(self, image):
        boxed_image = letterbox_image(image, tuple(reversed(self.model_input_shape)))

        image_data = np.expand_dims(boxed_image, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.shape[0], image.shape[1]],
                K.learning_phase(): 0
            })

        return out_boxes, out_scores, out_classes

def detect_video(yolo):
    import pyrealsense2 as rs
    try:
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()
        # Create a context object. This object owns the handles to all connected realsense devices
        cfg = rs.config()
        cfg.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8,
                          60)
        pipeline = rs.pipeline()
        pipeline.start(cfg)
        while True:
            # Create a pipeline object. This object configures the streaming camera and owns it's handle
            frames = pipeline.wait_for_frames()
            image = frames.get_infrared_frame()
            if not image:
                continue
            image = np.asanyarray(image.get_data())
            if len(image.shape) > 2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = yolo.detect_image(image)
            result = np.asarray(image)
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pass
