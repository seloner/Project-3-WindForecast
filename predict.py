import keras
import numpy as np
import pandas as pd
import sys
from mape import calculate_mape
from keras import layers, optimizers, losses, metrics
from keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
if(sys.argv[1] == "-i"):
    path = sys.argv[2]
else:
    print("Give input file \n")
    quit()
# Initializes the model
model = load_model("./WindDenseNN.h5")
model.summary()
model.compile(optimizer="adam", loss='mse',metrics=['MAPE'])
# read data for prediction
data = pd.read_csv(path)
# save labels
labels = data[data.columns[0]]
# drop label(first column)
input_data = data.drop(data.columns[0], axis=1)
# read actual data for error calculate
actual = pd.read_csv("./actual.csv")
# drop label(first column)
actual_data = actual.drop(data.columns[0], axis=1)
result = model.predict(input_data, batch_size=32)
# calculate errors
mae = mean_absolute_error(actual_data, result)
mse = mean_squared_error(actual_data, result)
mape=calculate_mape(actual_data,result)
# convert to data frames
df = pd.DataFrame(result)
df2 = pd.DataFrame(labels)
f = open('./predicted.csv', 'w')
error = "MAE:" + str(mae) + "\t"+'MAPE: '+str(mape)+"\t""MSE: "+str(mse)+"\n"
# write error statistics to file
f.write(error)
f.close()
# concat labels with predictions
csv = pd.concat([df2, df], axis=1)
# append to predicted file as csv
pd.DataFrame(csv).to_csv(
    "predicted.csv", index=None, header=None,  sep="\t", mode="a")

# mape=mean_absolute_percentage_error([1],[1])

