import tensorflow as tf
import numpy as np

model_path = "models/mnist784_16x8_1v2.keras"
model = tf.keras.models.load_model(model_path)

A = []
B = []


for i in range(1, len(model.layers)):
    print("Layer", i)
    A.append(model.layers[i].get_weights()[0])
    print("Weight Matrix")
    print(A[-1].T)
    B.append(model.layers[i].get_weights()[1])
    print("Weight Matrix")
    print(B[-1].T)
