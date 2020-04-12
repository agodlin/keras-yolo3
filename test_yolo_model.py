import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
from tqdm import tqdm
import numpy as np
import numpy
import cv2
from utils import yolo_utils

def test_iou(yolo, annotation_path):
    # val_images = yolo_utils.get_train_val_annotation_list(annotation_path)
    val_images = ['C:/work/yolov3-tf2/data/meme2.jpeg']
    tpr = 0
    err = 0
    err_score = 0
    err_iou = 0
    not_supported = 0
    total = 0
    for annotation_line in tqdm(val_images):
        file_path, rects, _ = yolo_utils.parse_line(annotation_line)
        try:
            color = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if len(color.shape) > 2:
                image = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            else:
                image = color
            image = image.astype(np.float32)
        except Exception as e:
            print('Open Error! continue to next ' + file_path)
            print(e)
            continue
        boxes, scores, out_classes = yolo.detect_image(image)

        if rects is None and (boxes is None or len(boxes) == 0):
            tpr += 1
            total += 1
            # continue
        elif rects is not None and (boxes is None or len(boxes) == 0):
            err += len(rects)
            total += len(rects)
            # continue
        elif rects is None:
            not_supported += 1
            total += 1
            # continue

        for i in range(len(boxes)):
            score = scores[i]
            if score < 0.5:
                err_score += 1
                total += 1
                continue
            top, left, bottom, right = boxes[i]
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))

            found = -1
            box = np.array([left, top, right, bottom])
            for j in range(len(rects)):
                rect = rects[j]
                iou = yolo_utils.bb_intersection_over_union(rect, box)

                if iou > 0.5:
                    found = j

            if found > -1:
                tpr += 1
                total += 1
            else:
                err_iou += 1
                total += 1

    print('not supported', not_supported)
    print('total tpr, err', tpr / total * 100, err / total * 100)
    print('score err, iou err', err_score / total * 100, err_iou / total * 100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str, default="C:/work/keras-yolo3/logs/003/ep003-loss11.749-val_loss11.061.h5"
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
        '--model_input_shape', type=tuple, default=(144, 144)
    )

    parser.add_argument(
        '--val_list', type=str, default="", help = "path to validation list"
    )
    FLAGS = parser.parse_args()
    anchors = yolo_utils.get_anchors(FLAGS.anchors_path)
    yolo = YOLO(FLAGS.model_path, anchors, FLAGS.num_classes, FLAGS.score_th, FLAGS.iou_th, FLAGS.model_input_shape)
    test_iou(yolo, FLAGS.val_list)
