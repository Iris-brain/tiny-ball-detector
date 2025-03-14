import time
import timeit
from typing import List, Tuple
import tensorflow as tf
import numpy as np
import keras


def time_model(model: keras.Model, shape: Tuple[int, ...], n: int) -> float:
    input_data = np.random.rand(*shape).astype(np.float32)

    input_tensor = tf.convert_to_tensor(input_data)

    @tf.function
    def predict_step(input_tensor):
        return model(input_tensor, training=False)

    _ = predict_step(input_tensor)

    time = timeit.timeit(lambda: predict_step(input_tensor), number=n) / n

    # Affichage du temps d'inférence
    print(f"Temps de prédiction du modèle (sans overhead Python) : {time:.6f} secondes")
    return time
