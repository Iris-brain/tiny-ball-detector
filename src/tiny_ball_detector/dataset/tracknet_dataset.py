from dataclasses import dataclass, field
import functools
from logging import Logger
from pathlib import Path
from typing import Any, Generator, Iterable, Tuple
import keras
import numpy as np
import tensorflow as tf


@dataclass
class TrackNetDatasetLoader:
    path: Path
    n_frames: int
    logger: Logger
    training: bool = False
    height = 1280
    width = 720
    dataset: tf.data.Dataset = field(init=False)
    size: int = field(init=False)
    variance: int = field(init=False)
    gaussian_kernel: np.ndarray = field(init=False)

    def __post_init__(self):
        self.dataset = tf.data.Dataset.from_generator(
            self.get_generator,
            output_types=(
                tf.float32,
                {
                    "file name": tf.string,
                    "visibility": tf.int32,
                    "x-coordinate": tf.int32,
                    "y-coordinate": tf.int32,
                    "status": tf.int32,
                },
            ),
            output_shapes=(
                (None, None, None, 3),
                {
                    "file name": (1,),
                    "visibility": (1,),
                    "x-coordinate": (1,),
                    "y-coordinate": (1,),
                    "status": (1,),
                },
            ),
        )

    def get_clips_and_labels_paths(self) -> dict[str, str]:
        clip_and_labels_paths_map = {
            str(p): str(p) + "/Label.csv" for p in list(self.path.glob("*/Clip*/"))
        }
        self.logger.error(f"📌 Number of clips found: {len(clip_and_labels_paths_map)}")

        return clip_and_labels_paths_map

    def concatenate(
        self, list_of_dataset: Iterable[tf.data.Dataset]
    ) -> tf.data.Dataset:
        return functools.reduce(
            lambda prev_dataset, current_dataset: prev_dataset.concatenate(
                current_dataset
            ),
            list_of_dataset,
        )

    def load_clip(self, clip_path: str) -> tf.data.Dataset:
        dataset: tf.data.Dataset = keras.utils.image_dataset_from_directory(
            directory=clip_path,
            labels=None,  # type: ignore
            label_mode=None,  # type: ignore
            image_size=(self.width, self.height),
            batch_size=None,  # type: ignore
            shuffle=False,
            pad_to_aspect_ratio=True,
            verbose=False,
        )
        return dataset

    def load_label(self, label_path: str):
        dataset = tf.data.experimental.make_csv_dataset(
            label_path,
            batch_size=1,  # Artificially small to make examples easier to show.
            num_epochs=1,
            ignore_errors=True,
            shuffle=False,
        )

        return dataset.enumerate().filter(skip(self.n_frames)).map(lambda _, x: x)

    def group_frames(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.batch(self.n_frames, drop_remainder=True)

    def get_generator(self) -> Generator[Tuple, Any, Any]:
        pairs = self.get_clips_and_labels_paths()

        for clip_path, label_path in pairs.items():
            clip_dataset = self.load_clip(clip_path=clip_path)
            clip_dataset = self.group_frames(dataset=clip_dataset)
            label_dataset = self.load_label(label_path)

            dataset = tf.data.Dataset.zip(clip_dataset, label_dataset)

            for sample in dataset:
                yield sample

    def __call__(self, size: int = 20, variance: int = 10) -> tf.data.Dataset:
        self.size = size
        self.variance = variance
        self.kernel_radius = int(size // 2)
        self.gaussian_kernel = create_gaussian_kernel(size, variance)
        dataset = self.dataset.apply(self.transform_coordinates_to_mask)
        return dataset

    def transform_coordinates_to_mask(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        for i, j in ds.take(4):
            distance_to_top = int(
                self.kernel_radius - min(j["y-coordinate"][0], self.kernel_radius)
            )
            distance_to_bottom = int(
                self.kernel_radius
                - min(self.height - j["y-coordinate"][0], self.kernel_radius)
            )
            distance_to_left = int(
                self.kernel_radius - min(j["x-coordinate"][0], self.kernel_radius)
            )
            distance_to_right = int(
                self.kernel_radius
                - min(self.width - j["x-coordinate"][0], self.kernel_radius)
            )
            print(
                self.kernel_radius,
                j["y-coordinate"],
                j["x-coordinate"],
                distance_to_top,
                distance_to_bottom,
                distance_to_left,
                distance_to_right,
                self.gaussian_kernel[
                    distance_to_left : -(distance_to_right + 1),
                    distance_to_top : -(distance_to_bottom + 1),
                ],
            )

        return ds


def skip(n_frames: int):
    def _skip(i: int, _):
        return tf.equal(i % n_frames, 2)

    return _skip


def gaussian_kernel(size: int, variance: int) -> np.ndarray:
    x, y = np.mgrid[-size : size + 1, -size : size + 1]
    g = np.exp(-(x**2 + y**2) / float(2 * variance))
    return g


def create_gaussian_kernel(size: int, variance: int) -> np.ndarray:
    gaussian_kernel_array = gaussian_kernel(size, variance)
    gaussian_kernel_array = gaussian_kernel_array * 255
    gaussian_kernel_array = gaussian_kernel_array.astype(int)
    return gaussian_kernel_array
