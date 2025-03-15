import keras
from keras import layers


class ConvBlock(layers.Layer):
    def __init__(
        self,
        out_channels,
        kernel_size=3,
        stride=1,
        use_bias=True,
        data_format="channels_last",
    ):
        super().__init__()
        self.block = keras.Sequential(
            [
                layers.Conv2D(
                    filters=out_channels,
                    kernel_size=kernel_size,
                    strides=stride,
                    padding="same",
                    use_bias=use_bias,
                    data_format=data_format,
                ),
                layers.ReLU(),
                layers.BatchNormalization(),
            ]
        )

    def build(self, input_shape):
        self.block.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        return self.block(x)
