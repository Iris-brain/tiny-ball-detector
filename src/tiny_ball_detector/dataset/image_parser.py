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
        self, filename: str, color_mode: ColorMode
    ) -> Tuple[tf.Tensor, tf.Tensor] | tf.Tensor:
        image = tf.io.read_file(filename)
        image = tf.io.decode_jpeg(image, channels=color_mode.value)
        image = tf.image.resize_with_pad(image, self.image_size[0], self.image_size[1])
        return image
