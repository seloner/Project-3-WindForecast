import keras
import numpy as np
import pandas as pd
import sys
from keras import layers, optimizers, losses, metrics
from keras.models import load_model
given_model = load_model("./WindDenseNN.h5")
weights = given_model.layers[0].get_weights()
model = keras.Sequential()
model.compile(optimizer=optimizers.RMSprop(
    0.01), loss=losses.CategoricalCrossentropy(), metrics=[metrics.CategoricalAccuracy()])
model.add(layers.Dense(64, activation="softmax"))
print(weights)
# model.layers[0].set_weights(weights)
# print(weights)
# model.summary()
# print(weights)
