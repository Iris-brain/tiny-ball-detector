import keras
from keras import Model
from keras.api.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Input,
    MaxPooling2D,
)
import tensorflow as tf
from .conv_bloc import ConvBlock


def gridTrackNet(imgs_per_instance, input_height, input_width):

    imgs_input = Input(shape=(imgs_per_instance * 3, input_height, input_width))

    x = Conv2D(
        64,
        (3, 3),
        kernel_initializer="random_uniform",
        padding="same",
        data_format="channels_first",
    )(imgs_input)
    x = (Activation("relu"))(x)
    x = (BatchNormalization())(x)

    x = Conv2D(
        64,
        (3, 3),
        kernel_initializer="random_uniform",
        padding="same",
        data_format="channels_first",
    )(x)
    x = (Activation("relu"))(x)
    x = (BatchNormalization())(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first")(x)

    x = Conv2D(
        128,
        (3, 3),
        kernel_initializer="random_uniform",
        padding="same",
        data_format="channels_first",
    )(x)
    x = (Activation("relu"))(x)
    x = (BatchNormalization())(x)

    x = Conv2D(
        128,
        (3, 3),
        kernel_initializer="random_uniform",
        padding="same",
        data_format="channels_first",
    )(x)
    x = (Activation("relu"))(x)
    x = (BatchNormalization())(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first")(x)

    x = Conv2D(
        256,
        (3, 3),
        kernel_initializer="random_uniform",
        padding="same",
        data_format="channels_first",
    )(x)
    x = (Activation("relu"))(x)
    x = (BatchNormalization())(x)

    x = Conv2D(
        256,
        (3, 3),
        kernel_initializer="random_uniform",
        padding="same",
        data_format="channels_first",
    )(x)
    x = (Activation("relu"))(x)
    x = (BatchNormalization())(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first")(x)

    x = (
        Conv2D(
            256,
            (3, 3),
            kernel_initializer="random_uniform",
            padding="same",
            data_format="channels_first",
        )
    )(x)
    x = (Activation("relu"))(x)
    x = (BatchNormalization())(x)

    x = (
        Conv2D(
            256,
            (3, 3),
            kernel_initializer="random_uniform",
            padding="same",
            data_format="channels_first",
        )
    )(x)
    x = (Activation("relu"))(x)
    x = (BatchNormalization())(x)

    x = (
        Conv2D(
            256,
            (3, 3),
            kernel_initializer="random_uniform",
            padding="same",
            data_format="channels_first",
        )
    )(x)
    x = (Activation("relu"))(x)
    x = (BatchNormalization())(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first")(x)

    x = (
        Conv2D(
            512,
            (3, 3),
            kernel_initializer="random_uniform",
            padding="same",
            data_format="channels_first",
        )
    )(x)
    x = (Activation("relu"))(x)
    x = (BatchNormalization())(x)

    x = (
        Conv2D(
            512,
            (3, 3),
            kernel_initializer="random_uniform",
            padding="same",
            data_format="channels_first",
        )
    )(x)
    x = (Activation("relu"))(x)
    x = (BatchNormalization())(x)

    x = (
        Conv2D(
            512,
            (3, 3),
            kernel_initializer="random_uniform",
            padding="same",
            data_format="channels_first",
        )
    )(x)
    x = (Activation("relu"))(x)
    x = (BatchNormalization())(x)

    x = (
        Conv2D(
            imgs_per_instance * 3,
            (3, 3),
            kernel_initializer="random_uniform",
            padding="same",
            data_format="channels_first",
        )
    )(x)
    output = (Activation("sigmoid"))(x)

    model = Model(imgs_input, output)

    return model


class GridTrackNet(Model):
    def __init__(self, imgs_per_instance, input_height, input_width):
        super(GridTrackNet, self).__init__()

        # Define layers similarly to the functional API version
        self.conv1 = ConvBlock(64)
        self.conv2 = ConvBlock(64)
        self.pool1 = keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=(2, 2), data_format="channels_first"
        )

        self.conv3 = ConvBlock(128)
        self.conv4 = ConvBlock(128)
        self.pool2 = keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=(2, 2), data_format="channels_first"
        )

        self.conv5 = ConvBlock(256)
        self.conv6 = ConvBlock(256)
        self.conv7 = ConvBlock(256)
        self.pool3 = keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=(2, 2), data_format="channels_first"
        )

        self.conv8 = ConvBlock(512)
        self.conv9 = ConvBlock(512)
        self.conv10 = ConvBlock(512)

        self.conv11 = ConvBlock(
            imgs_per_instance * 3
        )  # Output channels (as in the last Conv2D layer)

        self.softmax = keras.layers.Activation("sigmoid")

    def build(self, input_shape: tf.TensorShape):
        # Build the model by passing a sample input through the model layers
        x: tf.Tensor = keras.layers.Input(shape=input_shape[1:])  # type: ignore
        self.call(x)
        super().build(input_shape)

    def call(self, x: tf.Tensor, training=False):
        # Pass the input through the layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)

        x = self.conv11(x)  # Last convolutional block
        out = self.softmax(x)
        return out
