from keras import initializers
from keras import regularizers
from keras import constraints

from keras import backend as K
from keras.engine import InputSpec
import numpy as np
from keras.models import load_model
from tensorflow.python.framework import graph_util
import tensorflow as tf
import os

def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""

    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh

    return box_xy, box_wh, box_confidence, box_class_probs


def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def convert_model_to_tf(keras_model_path, output_path, custom_objects=None):
    """
    convert a Keras model (saved as .h5 file) to a TensorFlow graph (saved as .pb file)
    :param keras_model_path: a path to a .h5 file (string)
    :param output_path: a target path to save the .pb file (string)
    :param custom_objects: optional dictionary mapping names to custom classes or functions to be
            considered during deserialization. for example:
            custom_objects={'Landmarks2HeatMaps': Landmarks2HeatMaps, 'FixedConv2D': FixedConv2D})
    """
    # Validate output file name
    
    # Freeze Keras model
    K.set_learning_phase(0)
    print(os.path.exists(keras_model_path))

    d = {'yolo_head':yolo_head, 'box_iou':box_iou}
    custom_objects = d
    
    net_model = load_model(keras_model_path, custom_objects=custom_objects)

    outputs = [node.op.name for node in net_model.outputs]
    inputs = [node.op.name for node in net_model.inputs]


    sess_k = K.get_session()
    with sess_k.as_default():
        constant_graph = graph_util.convert_variables_to_constants(
            sess_k, sess_k.graph.as_graph_def(), outputs)
        
        with tf.gfile.GFile(output_path, "wb") as f:
            f.write(constant_graph.SerializeToString())

    K.clear_session()
    print('Saved the frozen model at: ', output_path)
    print("Input layers: {}".format(inputs))
    print("Output layers: {}".format(outputs))

convert_model_to_tf('/root/keras-yolo3/logs/000/trained_final.h5', '/root/keras-yolo3/logs/000/trained_final.pb')
