from dataclasses import dataclass
from enum import Enum
from functools import reduce
import os
from typing import Tuple

import tensorflow as tf


class ColorMode(Enum):
    GRAYSCALE = 1
    RGB = 3


@dataclass
class ImageParser:
    image_size: Tuple[int, int]

    def __call__(
        self, filename: str, color_mode: ColorMode, dtype=tf.float32
    ) -> tf.Tensor:
        print("DDDDDDD")
        image = tf.io.read_file(filename)
        image = tf.io.decode_jpeg(image, channels=color_mode.value)
        print(image.shape)
        image = tf.image.convert_image_dtype(image, dtype)
        image = tf.image.resize_with_pad(image, self.image_size[0], self.image_size[1])
        return image
