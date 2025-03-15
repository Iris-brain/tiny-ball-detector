from dataclasses import dataclass, field
import functools
from logging import Logger
import os
from pathlib import Path
from typing import Any, Generator, OrderedDict, Tuple
import numpy as np
import tensorflow as tf
from PIL import Image

from tiny_ball_detector.dataset.image_parser import ColorMode, ImageParser


@dataclass(kw_only=True)
class GameDataset:
    game: str
    clip: str
    clip_path: Path

    @property
    def dataset(self) -> tf.data.Dataset:
        return tf.data.Dataset.from_tensor_slices(
            [
                [str(path), str(path).replace("input", "output")]
                for path in self.clip_path.glob("*.jpg")
            ]
        )


@dataclass
class GridTrackNetDataset:
    path: Path
    n_frames: int
    logger: Logger
    training: bool = False
    height = 720
    width = 1280
    size: int = field(init=False)
    variance: int = field(init=False)
    gaussian_kernel: np.ndarray = field(init=False)

    def make_ground_truth_image(self, size: int = 20, variance: int = 10) -> None:
        self.size = size
        self.variance = variance
        self.gaussian_kernel = create_gaussian_kernel(size, variance)

        clip_path: str
        for clip_path, label in self.get_label():
            self.create_and_save_mask(label=label, clip_path=clip_path)

    def create_and_save_mask(self, label, clip_path: str) -> None:
        ball_is_visible = int(label["visibility"][0] != 0)
        y_coord = label["y-coordinate"][0]
        x_coord = label["x-coordinate"][0]

        distance_y_coor_to_top = min(y_coord, self.size)
        distance_y_coor_to_bottom = min(self.height - y_coord, self.size + 1)
        distance_x_coor_to_left = min(x_coord, self.size)
        distance_x_coor_to_right = min(self.width - x_coord, self.size + 1)

        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        new_gaussian_kernel = (
            ball_is_visible
            * self.gaussian_kernel[
                self.size
                - distance_y_coor_to_top : self.size
                + distance_y_coor_to_bottom,
                self.size
                - distance_x_coor_to_left : self.size
                + distance_x_coor_to_right,
            ]
        )

        mask[
            y_coord - distance_y_coor_to_top : y_coord + distance_y_coor_to_bottom,
            x_coord - distance_x_coor_to_left : x_coord + distance_x_coor_to_right,
        ] = new_gaussian_kernel

        self.save_grayscale_mask(mask, label, clip_path)

    def get_label(self) -> Generator[Tuple, Any, Any]:
        pairs = self.get_clips_and_labels_paths()

        for clip_path, label_path in pairs.items():
            label_dataset = self.load_label_csv(label_path)
            for label in label_dataset:
                yield clip_path, label

    def get_clips_and_labels_paths(self) -> dict[str, str]:
        clip_and_labels_paths_map = {
            str(p): str(p) + "/Label.csv"
            for p in sorted(list(self.path.glob("input/*/Clip*/")))
        }
        self.logger.error(f"📌 Number of clips found: {len(clip_and_labels_paths_map)}")

        return clip_and_labels_paths_map

    def load_label_csv(self, label_path: str):
        dataset = tf.data.experimental.make_csv_dataset(
            label_path,
            batch_size=1,
            num_epochs=1,
            ignore_errors=True,
            shuffle=False,
        )

        return dataset

    def save_grayscale_mask(self, mask: np.ndarray, label: OrderedDict, clip_path: str):
        im = Image.fromarray(mask, mode="L")

        file_name_str: str = label["file name"][0].numpy().decode("utf-8")
        output_path = clip_path.replace("input", "output")
        os.makedirs(output_path, exist_ok=True)

        im.save(f"{output_path}/{file_name_str}")

    @property
    def dataset(self) -> tf.data.Dataset:
        parser = ImageParser(image_size=(self.height, self.width))
        data = [
            GameDataset(
                game=clip_path.parts[-2],
                clip=clip_path.parts[-1],
                clip_path=clip_path,
            )
            .dataset.map(
                lambda x: (
                    parser(x[0], color_mode=ColorMode.RGB),
                    parser(x[1], color_mode=ColorMode.GRAYSCALE, dtype=tf.uint8),
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .batch(
                self.n_frames,
                drop_remainder=True,
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .map(self.transpose_reshape, num_parallel_calls=tf.data.AUTOTUNE)
            for clip_path in sorted(list(self.path.glob("input/*/Clip*/")))
        ]

        return functools.reduce(lambda x, y: x.concatenate(y), data)

    def transpose_reshape(self, image, label):
        image_transposed = tf.transpose(image, perm=[1, 2, 0, 3])
        image_final = tf.reshape(image_transposed, (self.height, self.width, 9))
        return image_final, label[[-1]]

    def group_frames(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.batch(self.n_frames, drop_remainder=True)


def gaussian_kernel(size: int, variance: int) -> np.ndarray:
    assert size % 2 == 0, "size should be a pair number"
    x, y = np.mgrid[-size : size + 1, -size : size + 1]
    g = np.exp(-(x**2 + y**2) / float(2 * variance))
    return g


def create_gaussian_kernel(size: int, variance: int) -> np.ndarray:
    gaussian_kernel_array = gaussian_kernel(size, variance)
    gaussian_kernel_array = gaussian_kernel_array * 255
    gaussian_kernel_array = gaussian_kernel_array.astype(int)
    return gaussian_kernel_array
