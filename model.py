import keras
import numpy as np
import pandas as pd
from keras import layers, optimizers, losses, metrics
from keras.models import load_model

# Initializes the model
model = load_model("./WindDenseNN.h5")
model.summary()
model.compile(optimizer=optimizers.RMSprop(
    0.01), loss=losses.CategoricalCrossentropy(), metrics=[metrics.CategoricalAccuracy()])
data = pd.read_csv('./nn_representations.csv')
# drop label(first column)
input_data = data.drop(data.columns[0], axis=1)
result = model.predict(input_data, batch_size=32)
print(result.shape)
print(result)
