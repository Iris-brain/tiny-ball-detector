import time
import tensorflow as tf
import numpy as np
import keras


def time_model(model: keras.Model) -> float:
    # Création d'une entrée factice (batch de 1 image 224x224x3)
    input_data = np.random.rand(1, *model.input_spec).astype(np.float32)

    # Conversion en tf.Tensor pour éviter l'overhead numpy -> tensorflow
    input_tensor = tf.convert_to_tensor(input_data)

    # Fonction compilée pour exécuter le modèle plus efficacement
    @tf.function
    def predict_step(input_tensor):
        return model(input_tensor, training=False)

    # Exécution une première fois pour chauffer le modèle (évite la latence initiale)
    _ = predict_step(input_tensor)

    # Mesure du temps en forçant l'exécution complète avec `.numpy()`
    start_time = time.perf_counter()
    predictions = predict_step(input_tensor)
    tf.test.experimental.sync_devices()  # Force l'exécution immédiate (utile si plusieurs devices)
    _ = predictions.numpy()  # Force le GPU à finir les calculs # type: ignore
    end_time = time.perf_counter()

    # Affichage du temps d'inférence
    print(
        f"Temps de prédiction du modèle (sans overhead Python) : {end_time - start_time:.6f} secondes"
    )
    return end_time - start_time
