import argparse
from yolo import YOLO, detect_video
import numpy as np
import cv2
import data_io
import mtcnn_pp
import os
from utils import yolo_utils

def detect_img(yolo, fp):
    images_types = ['png', 'jpg']
    images = [os.path.join(fp, name) for name in os.listdir(fp)]
    for img in images:
        print(os.path.basename(img))
        if img.endswith('w10'):
            image = data_io.read_w10(img)
            image = mtcnn_pp.mtcnn_preprocess(image, 1)
        elif img.split('.')[-1] in images_types:
            image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        else:
            continue
        # image = image[:, 308:716+308]
        yolo.detect_image(image)
        color = image.astype(np.uint8)
        color = cv2.cvtColor(color, cv2.COLOR_GRAY2RGB)

        out_boxes, out_scores, out_classes = yolo.detect_image(image)
        for i in range(len(out_boxes)):
            if out_scores[i] < 0.5:
                continue
            top, left, bottom, right = out_boxes[i]
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))
            box = np.array([left, top, right, bottom])
            print(box, right-left, bottom-top)
            cv2.rectangle(color, tuple(box[:2].astype(np.int)), tuple(box[2:4].astype(np.int)),
                          (0, 255, 0), 3)
        cv2.imshow('', color)
        cv2.waitKey()
    yolo.close_session()
    
FLAGS = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str, default="model_data/v2_176_1/trained_weights_final.h5"
    )

    parser.add_argument(
        '--anchors_path', type=str, default="model_data/face_anchors.txt"
    )

    parser.add_argument(
        '--num_classes', type=int, default=1
    )

    parser.add_argument(
        '--score_th', type=float, default=0.5
    )

    parser.add_argument(
        '--iou_th', type=int, default=0.5
    )

    parser.add_argument(
        '--model_input_shape', type=tuple, default=(176, 176)
    )

    parser.add_argument(
        '--image', type=str, default="", help = "images folder input path"
    )
    FLAGS = parser.parse_args()
    anchors = yolo_utils.get_anchors(FLAGS.anchors_path)
    yolo = YOLO(FLAGS.model_path, anchors, FLAGS.num_classes, FLAGS.score_th, FLAGS.iou_th, FLAGS.model_input_shape)
    if FLAGS.image:
        detect_img(yolo, FLAGS.image)
    else:
        detect_video(yolo)
