import tensorflow as tf
import numpy as np

model_path = '/home/haolin/workspace/results/model_extraction/checkpoints/tf/carlini/dense/10_8x2/linear/20241122-152802/model.keras'
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
