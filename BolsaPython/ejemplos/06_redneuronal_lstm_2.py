import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import *
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

datos_entrada = pd.read_csv("/home/carloslinux/Desktop/GIT_BOLSA/BolsaPython/ejemplos/06_redneuronal_lstm_TSLA.csv")
print("Entrada - Numero de filas y columnas:", datos_entrada.shape)
print(datos_entrada.head(5))
numfilas = datos_entrada.shape[0]
numfilastrain = 800
numfilastest = numfilas-numfilastrain

print("TRAIN y TEST...")
training_set = datos_entrada.iloc[:numfilastrain, 1:2].values  # En este caso elegimos que la entrada solo tenga una feature: [459x1]
test_set = datos_entrada.iloc[numfilastrain:, 1:2].values

print("NORMALIZAR entrada (max-min scaler) with time lag of 1 day (lag 1)...")
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 time-steps and 1 output
X_train = []
y_train = []
lstm_ventana = 60
for i in range(lstm_ventana, numfilastrain):
    X_train.append(training_set_scaled[i-lstm_ventana:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # Sus dimensiones son: (numfilastrain - lstm_ventana, lstm_ventana, 1)
print("Entrada - Tras reshape:", X_train.shape)
print(X_train)

########################### MODELO: LSTM #################################
# LSTM with 50 neurons and 4 hidden layers.
# Finally, we will assign 1 neuron in the output layer for predicting the normalized stock price.
# We will use the MSE loss function and the Adam stochastic gradient descent optimizer.
lstm_numneuronas_internas=50
lstm_numneuronas_capafinal=1
model = Sequential()  # Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units=lstm_numneuronas_internas, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))  # Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units=lstm_numneuronas_internas, return_sequences=True))
model.add(Dropout(0.2))  # Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units=lstm_numneuronas_internas, return_sequences=True))
model.add(Dropout(0.2))  # Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units=lstm_numneuronas_internas))
model.add(Dropout(0.2))  # Adding the output layer
model.add(Dense(units=lstm_numneuronas_capafinal))

# Compilando la RNN
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenamiento de la RNN (red neuronal recursiva) con el dataset TRAIN
lstm_epochs = 20  # 100
model.fit(X_train, y_train, epochs=lstm_epochs, batch_size=32)

############################################################
# Getting the predicted stock price of 2017
dataset_train = datos_entrada.iloc[:numfilastrain, 1:2]
dataset_test = datos_entrada.iloc[numfilastrain:, 1:2]
dataset_total = pd.concat((dataset_train, dataset_test), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - lstm_ventana:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(lstm_ventana, numfilastest + lstm_ventana):
    X_test.append(inputs[i-lstm_ventana:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  # Dimensiones: (numfilastest, lstm_ventana, 1)
print(X_test.shape)

############## Prediccion: usando el dataset TEST #################
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

######################### VISUALIZACION ##############################
plt.plot(datos_entrada.loc[numfilastrain:, "Date"], dataset_test.values, color="red", label="Real TESLA Stock Price")
plt.plot(datos_entrada.loc[numfilastrain:, "Date"], predicted_stock_price, color="blue", label="Predicted TESLA Stock Price")
plt.xticks(np.arange(0, numfilastest, 50))
plt.title('TESLA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('TESLA Stock Price')
plt.legend()
plt.show()
