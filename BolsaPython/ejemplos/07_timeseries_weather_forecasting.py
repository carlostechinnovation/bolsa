# WEB: https://keras.io/examples/timeseries/timeseries_weather_forecasting/

# Instalar: Python 3.8
# Desde la consola de py38 (en Pycharm):
# pip install tf-nightly
# pip install tensorflow-gpu

"""
Title: Timeseries forecasting for weather prediction
Authors: [Prabhanshu Attri](https://prabhanshu.com/github), [Yashika Sharma](https://github.com/yashika51), [Kristi Takach](https://github.com/ktakattack), [Falak Shah](https://github.com/falaktheoptimist)
Date created: 2020/06/23
Last modified: 2020/07/20
Description: This notebook demonstrates how to do timeseries forecasting using a LSTM model.
"""

"""
## Setup
This example requires TensorFlow 2.3 or higher.
"""
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.models import model_from_yaml
from keras import backend as K
import sys

print("INICIO")

###### PARAMETROS DE ENTRADA ###########
dir_subgrupo = sys.argv[1]
#########################################

"""
## Climate Data Time-Series

We will be using Jena Climate dataset recorded by the
[Max Planck Institute for Biogeochemistry](https://www.bgc-jena.mpg.de/wetter/).
The dataset consists of 14 features such as temperature, pressure, humidity etc, recorded once per
10 minutes.

**Location**: Weather Station, Max Planck Institute for Biogeochemistry
in Jena, Germany

**Time-frame Considered**: Jan 10, 2009 - December 31, 2016


The table below shows the column names, their value formats, and their description.

Index| Features      |Format             |Description
-----|---------------|-------------------|-----------------------
1    |Date Time      |01.01.2009 00:10:00|Date-time reference
2    |p (mbar)       |996.52             |The pascal SI derived unit of pressure used to quantify internal pressure. Meteorological reports typically state atmospheric pressure in millibars.
3    |T (degC)       |-8.02              |Temperature in Celsius
4    |Tpot (K)       |265.4              |Temperature in Kelvin
5    |Tdew (degC)    |-8.9               |Temperature in Celsius relative to humidity. Dew Point is a measure of the absolute amount of water in the air, the DP is the temperature at which the air cannot hold all the moisture in it and water condenses.
6    |rh (%)         |93.3               |Relative Humidity is a measure of how saturated the air is with water vapor, the %RH determines the amount of water contained within collection objects.
7    |VPmax (mbar)   |3.33               |Saturation vapor pressure
8    |VPact (mbar)   |3.11               |Vapor pressure
9    |VPdef (mbar)   |0.22               |Vapor pressure deficit
10   |sh (g/kg)      |1.94               |Specific humidity
11   |H2OC (mmol/mol)|3.12               |Water vapor concentration
12   |rho (g/m ** 3) |1307.75            |Airtight
13   |wv (m/s)       |1.03               |Wind speed
14   |max. wv (m/s)  |1.75               |Maximum wind speed
15   |wd (deg)       |152.3              |Wind direction in degrees
"""

from zipfile import ZipFile
import os

uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
zip_path = keras.utils.get_file(origin=uri, fname="jena_climate_2009_2016.csv.zip")
zip_file = ZipFile(zip_path)
zip_file.extractall()
csv_path = "jena_climate_2009_2016.csv"

df = pd.read_csv(csv_path)
print("ORIGINAL:"); print(df.head(3))

"""
## Raw Data Visualization

To give us a sense of the data we are working with, each feature has been plotted below.
This shows the distinct pattern of each feature over the time period from 2009 to 2016.
It also shows where anomalies are present, which will be addressed during normalization.
"""
titles = ["Pressure", "Temperature", "Temperature in Kelvin", "Temperature (dew point)", "Relative Humidity", "Saturation vapor pressure",
          "Vapor pressure", "Vapor pressure deficit", "Specific humidity", "Water vapor concentration", "Airtight",
          "Wind speed", "Maximum wind speed", "Wind direction in degrees",]

feature_keys = ["p (mbar)",  "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)", "VPmax (mbar)",
    "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)", "H2OC (mmol/mol)", "rho (g/m**3)", "wv (m/s)",
    "max. wv (m/s)", "wd (deg)",]

colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan",]

date_time_key = "Date Time"

def show_raw_visualization(data):
    time_data = data[date_time_key]
    fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k")
    for i in range(len(feature_keys)):
        key = feature_keys[i]
        c = colors[i % (len(colors))]
        t_data = data[key]
        t_data.index = time_data
        t_data.head()
        ax = t_data.plot(ax=axes[i // 2, i % 2], color=c, title="{} - {}".format(titles[i], key), rot=25,)
        ax.legend([titles[i]])
    plt.tight_layout()

#show_raw_visualization(df)

"""
This heat map shows the correlation between different features.
"""
def show_heatmap(data):
    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.show()

#show_heatmap(df)

"""
## Data Preprocessing

Here we are picking ~300,000 data points for training. Observation is recorded every
10 mins, that means 6 times per hour. We will resample one point per hour since no
drastic change is expected within 60 minutes. We do this via the `sampling_rate`
argument in `timeseries_dataset_from_array` utility.

We are tracking data from past 720 timestamps (720/6=120 hours). 
This data will be used to PREDICT the temperature after 72 timestamps (76/6=12 hours) (TARGET!!!!!!)

Since every feature has values with
varying ranges, we do normalization to confine feature values to a range of `[0, 1]` before
training a neural network.
We do this by subtracting the mean and dividing by the standard deviation of each feature.

71.5 % of the data will be used to train the model, i.e. 300,693 rows. `split_fraction` can
be changed to alter this percentage.

The model is shown data for first 5 days i.e. 720 observations, that are sampled every
hour. The temperature after 72 (12 hours * 6 observation per hour) observation will be
used as a label.
"""
split_fraction = 0.75
train_split = int(split_fraction * int(df.shape[0]))
step = 6

learning_rate = 0.001
batch_size = 8192  # MEMORIA
epochs = 1 #10
pathModeloKerasLSTM = dir_subgrupo + "keras_lstm.modelo"

def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std

"""
We can see from the correlation heatmap, few parameters like Relative Humidity and
Specific Humidity are redundant. Hence we will be using select features, not all.
"""
print("The selected parameters are:", ", ".join([titles[i] for i in [0, 1, 5, 7, 8, 10, 11]]),)
selected_features = [feature_keys[i] for i in [0, 1, 5, 7, 8, 10, 11]]
features = df[selected_features]
print("REDUCIDO:"); print(features.head(3))
features.index = df[date_time_key]

features = normalize(features.values, train_split)
features = pd.DataFrame(features)
print("REDUCIDO NORMALIZADO:"); print(features.head(3))

train_data = features.loc[0:(train_split - 1)]
val_data = features.loc[train_split:]
###############################################################################################################

"""
# Training dataset

The training dataset labels (los TARGETs) starts from the 792nd observation (720 + 72). ==> Para los hechos X en indice=0, su target Y sucedió en el futuro en indice=792 
"""
past = 720
future = 72
start = past + future
end = start + train_split

x_train = train_data[[i for i in range(train_data.shape[1])]].values
print("x_train:"); print(x_train[(start+1):(start+5)])
y_train = features.iloc[start:end][[1]]  # TARGET
print("y_train:"); print(y_train[1:5])  # TARGET OBSERVAR EL INDICE DE ESTE DATAFRAME. EMPIEZA POR 793!!!
sequence_length = int(past / step)

"""
The `timeseries_dataset_from_array` function takes in a sequence of data-points gathered at
equal intervals, along with time series parameters such as length of the
sequences/windows, spacing between two sequence/windows, etc., to produce batches of
sub-timeseries inputs and targets sampled from the main timeseries.
"""
print("Calculando dataset_train (MATRIZ)...")
dataset_train = keras.preprocessing.timeseries_dataset_from_array(x_train, y_train, sequence_length=sequence_length, sampling_rate=step, batch_size=batch_size,)

"""
## Validation dataset

The validation dataset must not contain the LAST 792 rows as we won't have label data (TARGET) for
those records, hence 792 must be subtracted from the end of the data.

The validation label dataset must start from 792 after train_split, hence we must add
past + future (792) to label_start.
"""
x_end = len(val_data) - past - future
label_start = train_split + past + future
val_numcolumnas = val_data.shape[1]
x_val = val_data.iloc[:x_end][[i for i in range(val_numcolumnas)]].values
y_val = features.iloc[label_start:][[1]]

print("Calculando dataset_val (MATRIZ)...")
dataset_val = keras.preprocessing.timeseries_dataset_from_array(x_val, y_val, sequence_length=sequence_length, sampling_rate=step, batch_size=batch_size,)

for batch in dataset_train.take(1):
    inputs, targets = batch

print("Input shape:", inputs.numpy().shape)
print("Target shape:", targets.numpy().shape)

"""
## Training
"""
print("Training - MODELO...")
inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = keras.layers.LSTM(32)(inputs)
outputs = keras.layers.Dense(1)(lstm_out)

# EXPLICACION: https://www.tensorflow.org/api_docs/python/tf/keras/Model
modelo = keras.Model(inputs=inputs, outputs=outputs)
modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=['accuracy'],
               loss_weights=None, weighted_metrics=None, run_eagerly=None)
modelo.summary()

"""
We'll use the `ModelCheckpoint` callback to regularly save checkpoints, and
the `EarlyStopping` callback to interrupt training when the validation loss
is not longer improving.
"""
path_checkpoint = "model_checkpoint.h5"
es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

print("Training - ModelCheckpoint...")
modelckpt_callback = keras.callbacks.ModelCheckpoint(monitor="val_loss", filepath=path_checkpoint, verbose=1, save_weights_only=True, save_best_only=True,)

print("Training - fit...")
history = modelo.fit(dataset_train, epochs=epochs, validation_data=dataset_val, callbacks=[es_callback, modelckpt_callback],)

print("Guardando modelo KERAS en disco...")
modelo.save(pathModeloKerasLSTM)



"""
We can visualize the loss with the function below. After one point, the loss stops
decreasing.
"""
def visualize_loss(history, title):
    print("Visualize_loss...")
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

visualize_loss(history, "Training and Validation Loss")

"""
## Prediction

The trained model above is now able to make predictions for 5 sets of values from
VALIDATION set.
"""
def show_plot(plot_data, delta, title):
    print("show_plot...")
    labels = ["History", "True Future", "Model Prediction"]
    marker = [".-", "rx", "go"]
    time_steps = list(range(-(plot_data[0].shape[0]), 0))
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, val in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel("Time-Step")
    plt.show()
    return


################################# Obtener metricas (VALIDATION) #########################
# print("Leyendo modelo KERAS...")
modelo_leido = keras.models.load_model(pathModeloKerasLSTM)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# evaluate the model
#loss, accuracy, f1_score, precision, recall = modelo_leido.evaluate(x_val, y_val, verbose=0)

######################################## FUTURO (uso validation, pero deberia usar X de futuro) ###################################

# Datos de validacion: X, Y
for batch in dataset_val.take(1):
    inputs_val, targets_val = batch

y_val_predicho = modelo_leido.predict(x=inputs_val, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)

#Predecir y pintar 3 velas del dataset VAL
for x, y in dataset_val.take(3):
    ejex = x[0][:, 1].numpy()
    yreal = y[0].numpy()
    ypredicho = modelo_leido.predict(x)[0]
    delta = 12
    title = "Single Step Prediction"
    show_plot([ejex, yreal, ypredicho], delta, title)

print("FIN")

