"""MobileNet v2 models for Keras.

# Reference
- [Inverted Residuals and Linear Bottlenecks Mobile Networks for
   Classification, Detection and Segmentation]
   (https://arxiv.org/abs/1801.04381)
"""

import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

def _conv_block(inputs, filters, kernel, strides, trainable=True):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.

    # Returns
        Output tensor.
    """

    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    x = Conv2D(filters, kernel, padding='same', strides=strides, **darknet_conv_kwargs)(inputs)
    x = BatchNormalization()(x)
    return LeakyReLU(alpha=0.1)(x)


def _bottleneck(inputs, filters, kernel, t, s, r=False, trainable=True):
    """Bottleneck
    This function defines a basic bottleneck structure.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        r: Boolean, Whether to use the residuals.

    # Returns
        Output tensor.
    """

    tchannel = K.int_shape(inputs)[-1]*t

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1), trainable=trainable)

    x = _conv_block(x, filters, (1, 1), (s, s), trainable=trainable)

    return x

def _bottleneck2(inputs, filters, kernel, t, s, r=False, trainable=True):
    """Bottleneck
    This function defines a basic bottleneck structure.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        r: Boolean, Whether to use the residuals.

    # Returns
        Output tensor.
    """

    x = _conv_block(inputs, filters, kernel, (s, s), trainable=trainable)

    return x

def _inverted_residual_block(inputs, filters, kernel, t, strides, n, trainable=True):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
        Output tensor.
    """

    x = _bottleneck(inputs, filters, kernel, t, strides, trainable=trainable)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, r=True, trainable=trainable)

    return x

def _inverted_residual_block2(inputs, filters, kernel, t, strides, n, trainable=True):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
        Output tensor.
    """
    x = _bottleneck2(inputs, filters, kernel, t, strides, trainable=trainable)

    x = _bottleneck2(x, filters, kernel, t, 1, r=True, trainable=trainable)

    return x

def build_model_2(inputs, expansion=3, last_conv_block_size=1280, last_inverted=False):
    """MobileNetv2
    This function defines a MobileNetv2 architectures.

    # Arguments
        input_shape: An integer or tuple/list of 3 integers, shape
            of input tensor.
        k: Integer, number of classes.
    # Returns
        MobileNetv2 model.
    """

    trainable = True
    x = _conv_block(inputs, 16, (3, 3), strides=(2, 2), trainable=trainable)
    x = _inverted_residual_block2(x, 32, (3, 3), t=expansion, strides=1, n=2, trainable=trainable)
    x = _inverted_residual_block2(x, 64, (3, 3), t=expansion, strides=2, n=3, trainable=trainable)
    x = _inverted_residual_block2(x, 96, (3, 3), t=expansion, strides=2, n=4, trainable=trainable)
    x = _inverted_residual_block2(x, 128, (3, 3), t=expansion, strides=1, n=3, trainable=trainable)
    x = _inverted_residual_block2(x, 256, (3, 3), t=expansion, strides=2, n=3, trainable=trainable)
    # x = _inverted_residual_block2(x, 1024, (3, 3), t=expansion, strides=1, n=3, trainable=trainable)
    if last_inverted:
        x = _inverted_residual_block(x, 320, (3, 3), t=2, strides=1, n=1,trainable=trainable)
    if last_conv_block_size > 0:
        x = _conv_block(x, last_conv_block_size, (1, 1), strides=(1, 1), trainable=trainable)
    else:
        print('Skipping last convolution layer')

    return x