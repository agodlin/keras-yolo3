"""MobileNet v2 models for Keras.

# Reference
- [Inverted Residuals and Linear Bottlenecks Mobile Networks for
   Classification, Detection and Segmentation]
   (https://arxiv.org/abs/1801.04381)
"""

from keras.layers import Conv2D, BatchNormalization, add, DepthwiseConv2D, LeakyReLU
from keras import backend as K


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

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides, trainable=trainable)(inputs)
    x = BatchNormalization(axis=channel_axis, trainable=trainable)(x)
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

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    tchannel = K.int_shape(inputs)[channel_axis] * t

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1), trainable=trainable)

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same', trainable=trainable)(x)
    x = BatchNormalization(axis=channel_axis, trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', trainable=trainable)(x)
    x = BatchNormalization(axis=channel_axis, trainable=trainable)(x)

    if r:
        x = add([x, inputs])
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


def build_model(inputs, expansion=3, last_conv_block_size=1280, last_inverted=False):
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

    x = _conv_block(inputs, 32, (3, 3), strides=(2, 2), trainable=trainable)

    x = _inverted_residual_block(x, 16, (3, 3), t=1, strides=1, n=1, trainable=trainable)
    x = _inverted_residual_block(x, 24, (3, 3), t=expansion, strides=1, n=2, trainable=trainable)
    x = _inverted_residual_block(x, 32, (3, 3), t=expansion, strides=2, n=3, trainable=trainable)
    x = _inverted_residual_block(x, 64, (3, 3), t=expansion, strides=2, n=4, trainable=trainable)
    x = _inverted_residual_block(x, 96, (3, 3), t=expansion, strides=1, n=3, trainable=trainable)
    x = _inverted_residual_block(x, 160, (3, 3), t=expansion, strides=2, n=3, trainable=trainable)
    if last_inverted:
        x = _inverted_residual_block(x, 320, (3, 3), t=2, strides=1, n=1,trainable=trainable)
    if last_conv_block_size > 0:
        x = _conv_block(x, last_conv_block_size, (1, 1), strides=(1, 1), trainable=trainable)
    else:
        print('Skipping last convolution layer')

    return x
