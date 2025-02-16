import tensorflow as tf
import keras
from keras import layers


class ConvBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size=3, stride=1, use_bias=True):
        super().__init__()
        self.block = keras.Sequential(
            [
                layers.Conv2D(
                    filters=out_channels,
                    kernel_size=kernel_size,
                    strides=stride,
                    padding="same",
                    use_bias=use_bias,
                ),
                layers.ReLU(),
                layers.BatchNormalization(),
            ]
        )

    def call(self, x):
        return self.block(x)


class TrackerNet(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(out_channels=64)
        self.conv2 = ConvBlock(out_channels=64)
        self.pool1 = layers.MaxPool2D(strides=2)
        self.conv3 = ConvBlock(out_channels=128)
        self.conv4 = ConvBlock(out_channels=128)
        self.pool2 = layers.MaxPool2D(strides=2)
        self.conv5 = ConvBlock(out_channels=256)
        self.conv6 = ConvBlock(out_channels=256)
        self.conv7 = ConvBlock(out_channels=256)
        self.pool3 = layers.MaxPool2D(strides=2)
        self.conv8 = ConvBlock(out_channels=512)
        self.conv9 = ConvBlock(out_channels=512)
        self.conv10 = ConvBlock(out_channels=512)
        self.ups1 = layers.UpSampling2D(size=2)
        self.conv11 = ConvBlock(out_channels=256)
        self.conv12 = ConvBlock(out_channels=256)
        self.conv13 = ConvBlock(out_channels=256)
        self.ups2 = layers.UpSampling2D(size=2)
        self.conv14 = ConvBlock(out_channels=128)
        self.conv15 = ConvBlock(out_channels=128)
        self.ups3 = layers.UpSampling2D(size=2)
        self.conv16 = ConvBlock(out_channels=64)
        self.conv17 = ConvBlock(out_channels=64)
        self.conv18 = ConvBlock(out_channels=256)

        self.softmax = layers.Softmax()

    def call(self, x: tf.Tensor, training=False):
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
        x = self.ups1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.ups2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.ups3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        out = self.softmax(x)
        return out
