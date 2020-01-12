import keras
import numpy as np
import pandas as pd
import sys
from keras import layers, optimizers, losses, metrics
from keras.models import load_model
if(sys.argv[1] == "-i"):
    path = sys.argv[2]
else:
    print("Give input file \n")
    quit()
data = pd.read_csv(path)
input_data = data.drop(data.columns[0], axis=1)
labels = data[data.columns[0]]
given_model = load_model("./WindDenseNN.h5")
weights = given_model.layers[0].get_weights()
model = keras.Sequential()
model.compile(optimizer=optimizers.RMSprop(
    0.01), loss="mse", )
model.add(layers.Dense(64, activation="linear",input_shape=(128,)))
model.layers[0].set_weights(weights)
# model.evaluate(input_data, labels, batch_size=32)
result = model.predict(input_data ,batch_size=32)
df = pd.DataFrame(result)
df2 = pd.DataFrame(labels)
csv = pd.concat([df2, df], axis=1)
pd.DataFrame(csv).to_csv(
    "new_representation.csv", index=None, header=None,  sep="\t", mode="w")
