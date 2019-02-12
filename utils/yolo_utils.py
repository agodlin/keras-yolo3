import numpy as np

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def get_annotation_list(annotation_path):
    with open(annotation_path) as f:
        lines = f.readlines()
    print('Testing on', len(lines), 'images')
    return lines


def get_train_val_annotation_list(annotation_path, val_split=0.1):
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    val_images = lines[num_train:]
    print('Testing on', len(val_images), 'images')
    return val_images


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
    file_path = data[0]
    if len(data) == 1:
        return file_path, None, None

    rects = []
    for i in range(1, len(data)):
        rect_str, class_id = data[1].split(',')[:4], data[1].split(',')[-1]
        rect = list(map(int, rect_str))
        rects.append(rect)

    return file_path, np.array(rects), class_id