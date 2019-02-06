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

    d = {}
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

if __name__ == '__main__':
    convert_model_to_tf('logs/000/trained_final.h5', 'logs/000/trained_final.pb')


