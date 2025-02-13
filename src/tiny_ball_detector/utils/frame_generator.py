from dataclasses import dataclass
import functools
import itertools
from logging import Logger
import os
from pathlib import Path
import random
from typing import Iterable
import pandas as pd
import cv2
import math
import keras
import tensorflow as tf


@dataclass
class FrameGenerator:
    path: Path
    n_frames: int
    logger: Logger
    training: bool = False

    def get_clips_and_labels_paths(self) -> dict[str, str]:
        clip_and_labels_paths_map = {
            str(p): str(p) + "/Label.csv" for p in list(self.path.glob("*/Clip*/"))
        }
        self.logger.error(f"📌 Number of clips found: {len(clip_and_labels_paths_map)}")

        return clip_and_labels_paths_map

    def concatenate(self, list_of_dataset: Iterable[tf.data.Dataset]):
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
            image_size=(1280, 720),
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

    def group_frames(self, dataset: tf.data.Dataset):
        return dataset.batch(self.n_frames, drop_remainder=True)

    def __call__(self):
        pairs = self.get_clips_and_labels_paths()

        list_of_dataset = []
        for clip_path, label_path in pairs.items():
            clip_dataset = self.load_clip(clip_path=clip_path)
            clip_dataset = self.group_frames(dataset=clip_dataset)
            label_dataset = self.load_label(label_path)
            dataset = tf.data.Dataset.zip(clip_dataset, label_dataset)
            list_of_dataset.append(dataset)

        return self.concatenate(list_of_dataset).map(self.transform_coordinates_to_mask)

    @staticmethod
    def transform_coordinates_to_mask(x: tf.Tensor, label):
        shape = x.shape


def skip(n_frames: int):
    def _skip(i: int, _):
        return tf.equal(i % n_frames, 2)

    return _skip
