import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
from tqdm import tqdm
import numpy as np
import numpy
import cv2
def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()
    
FLAGS = None

def test_iou(yolo, annotation_path):
    val_images = get_annotation_list(annotation_path)
    tpr = 0
    err = 0
    err_score = 0
    err_iou = 0
    not_supported = 0
    for annotation_line in tqdm(val_images):
        file_path, rect, _ = parse_line(annotation_line)
        try:
            #image = Image.open(file_path)
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = image.astype(np.float32)
        except Exception as e:
            print('Open Error! continue to next ' + file_path)
            print(e)
            return
        out_boxes, out_scores, out_classes = yolo.detect_image2(image)
        if out_boxes is None:
            err += 1
            continue        
        if len(out_boxes) == 1 and rect is not None:
            score = out_scores[0]
            if score < 0.5:
                err_score+=1
                continue            

            top, left, bottom, right = out_boxes[0]
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))

            box = [left, top, right, bottom]
#            print(file_path, rect, box)
            iou = bb_intersection_over_union(rect, box)
            if iou < 0.5:
                err_iou += 1
            else:
                tpr+=1
        else:
            err+=1
    total = len(val_images)-not_supported
    print('not supported', not_supported)
    print('total tpr, err', tpr/total*100, err/total*100)
    print('score err, iou err', err_score/total*100, err_iou/total*100)
            
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def parse_line(image_line):
    data = image_line.strip().split()
    if len(data) > 2:
        return None,None,None
    file_path = data[0]
    
    if len(data) == 1:
        return file_path, None, None

    rect_str,class_id = data[1].split(',')[:4],data[1].split(',')[-1]
    
    rect = list(map(int, rect_str))
    return file_path, rect, class_id


def get_annotation_list(annotation_path):
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    val_images = lines[num_train:]
    print('Testing on', len(val_images), 'images')
    return val_images[:10]


if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors_path', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes_path', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )
    parser.add_argument(
        '--test_iou', default=''
    )
    
    FLAGS = parser.parse_args()

    if FLAGS.test_iou:
        test_iou(YOLO(**vars(FLAGS)), FLAGS.test_iou)        
    
    elif FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
