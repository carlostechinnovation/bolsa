#WEB: https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

# LSTM for international airline passengers problem with window regression framing
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
dataframe = read_csv('/home/carloslinux/Desktop/GIT_BOLSA/BolsaPython/ejemplos/05_redneuronal_pasajerosavion.csv', usecols=[1], engine='python')
dataset = dataframe.values  # En este caso la entrada  solo tiene una feature: [144x1]
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

########################## MODELO: LSTM REGRESSION WITH TIME STEPS ##################
# The network has a visible layer with 1 input, a hidden layer with 4 LSTM blocks or neurons, and an output layer that makes a single value prediction.
# The default sigmoid activation function is used for the LSTM blocks.
# The network is trained for 100 epochs and a batch size of 1 is used

# convert an array of values into a dataset matrix
def create_dataset_lstm(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

# reshape into X=t and Y=t+1
look_back = 5  # DIAS ATRAS COGIDOS
trainX, trainY = create_dataset_lstm(train, look_back)
testX, testY = create_dataset_lstm(test, look_back)

print("trainX: "); print(trainX)

# Preparar la entrada a LSTM con este formato: [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

print("trainX (tras reshape): "); print(trainX)

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(look_back, 1)))  # REGRESSION WITH TIME STEPS
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


######################## PLOT ###############################
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions

plt.plot(scaler.inverse_transform(dataset), label="ORIGINAL")
plt.plot(trainPredictPlot, label="trainpredicho")
plt.plot(testPredictPlot, label="testpredicho")
#plt.legend(loc="upper left")  #Leyenda
#plt.ylim(-1.5, 2.0)
plt.show()
#############################################################3