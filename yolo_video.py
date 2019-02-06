import argparse
from yolo import YOLO, detect_video
import numpy as np
import cv2
import data_io
import mtcnn_pp
import os

def detect_img(yolo, fp):
    images = [os.path.join(fp, name) for name in os.listdir(fp) if name.endswith('w10')]
    for img in images:
        print(os.path.basename(img))
        image = data_io.read_w10(img)
        image = mtcnn_pp.mtcnn_preprocess(image, 1)
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
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str, default=""
    )

    parser.add_argument(
        '--anchors_path', type=str, default=""
    )

    parser.add_argument(
        '--classes_path', type=str, default=""
    )

    parser.add_argument(
        '--image', type=str, default="", help = "images folder input path"
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", type=str, default="", help = "Video input path"
    )

    FLAGS = parser.parse_args()
    d = {}
    d.update(**vars(FLAGS))
    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**d), FLAGS.image)
    elif "input" in FLAGS:
        detect_video(YOLO(**d), FLAGS.input)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
