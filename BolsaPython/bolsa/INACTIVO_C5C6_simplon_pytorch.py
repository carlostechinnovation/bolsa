import os
from pathlib import Path
from random import sample, choice

from imblearn.under_sampling import TomekLinks
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV, RFE
from sklearn import metrics
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
import pickle
import warnings
import datetime
from sklearn.pipeline import make_pipeline
from sklearn.manifold import TSNE
import math
import sys
import numpy as np
from imblearn.combine import SMOTEENN
from sklearn.feature_selection import SelectKBest
from sklearn.utils.fixes import loguniform
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.utils import resample
from sklearn.calibration import CalibratedClassifierCV
import os.path
from sklearn.tree import export_graphviz
from subprocess import call

from xgboost import XGBClassifier
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

print("CUDA IS AVAILABLE???:", torch.cuda.is_available())

print((datetime.datetime.now()).strftime(
    "%Y%m%d_%H%M%S") + " **** CAPA 5  --> Selección de variables/ Reducción de dimensiones (para cada subgrupo) ****")
print("URL PCA: https://scikit-learn.org/stable/modules/unsupervised_reduction.html")
print("URL Feature Selection: https://scikit-learn.org/stable/modules/feature_selection.html")

##################################################################################################
print("PARAMETROS: ")
dir_subgrupo = sys.argv[1]
modoTiempo = sys.argv[2]
maxFeatReducidas = sys.argv[3]
maxFilasEntrada = sys.argv[4]
desplazamientoAntiguedad = sys.argv[5]

print("dir_subgrupo = %s" % dir_subgrupo)
print("modoTiempo = %s" % modoTiempo)
print("maxFeatReducidas = %s" % maxFeatReducidas)
print("maxFilasEntrada = %s" % maxFilasEntrada)

varianza = 0.92  # Variacion acumulada de las features PCA sobre el target
compatibleParaMuchasEmpresas = False  # Si hay muchas empresas, debo hacer ya el undersampling (en vez de capa 6)
global modoDebug;
modoDebug = False  # VARIABLE GLOBAL: En modo debug se pintan los dibujos. En otro caso, se evita calculo innecesario
global cv_todos;
cv_todos = 15  # CROSS_VALIDATION: número de iteraciones. Sirve para evitar el overfitting
global rfecv_step;
rfecv_step = 3  # Numero de features que va reduciendo en cada iteracion de RFE hasta encontrar el numero deseado
global dibujoBins;
dibujoBins = 20  # VARIABLE GLOBAL: al pintar los histogramas, define el número de barras posibles en las que se divide el eje X.
numTramos = 7  # Numero de tramos usado para tramificar las features dinámicas
pathCsvCompleto = dir_subgrupo + "COMPLETO.csv"
dir_subgrupo_img = dir_subgrupo + "img/"
pathCsvIntermedio = dir_subgrupo + "intermedio.csv"
pathCsvReducido = dir_subgrupo + "REDUCIDO.csv"
pathCsvFeaturesElegidas = dir_subgrupo + "FEATURES_ELEGIDAS_RFECV.csv"
pathModeloOutliers = (dir_subgrupo + "DETECTOR_OUTLIERS.tool").replace("futuro", "pasado")  # Siempre lo cojo del pasado
path_modelo_tramificador = (dir_subgrupo + "tramif/" + "TRAMIFICADOR").replace("futuro",
                                                                               "pasado")  # Siempre lo cojo del pasado
path_modelo_normalizador = (dir_subgrupo + "NORMALIZADOR.tool").replace("futuro",
                                                                        "pasado")  # Siempre lo cojo del pasado
path_indices_out_capa5 = (dir_subgrupo + "indices_out_capa5.indices")
path_modelo_reductor_features = (dir_subgrupo + "REDUCTOR.tool").replace("futuro",
                                                                         "pasado")  # Siempre lo cojo del pasado
path_modelo_pca = (dir_subgrupo + "PCA.tool").replace("futuro", "pasado")  # Siempre lo cojo del pasado
path_pesos_pca = (dir_subgrupo + "PCA_matriz.csv")

balancear = False  # No usar este balanceo, sino el de Luis (capa 6), que solo actúa en el dataset de train, evitando tocar test y validation

print("pathCsvCompleto = %s" % pathCsvCompleto)
print("dir_subgrupo_img = %s" % dir_subgrupo_img)
print("pathCsvReducido = %s" % pathCsvReducido)
print("pathModeloOutliers = %s" % pathModeloOutliers)
print("path_modelo_normalizador = %s" % path_modelo_normalizador)
print("path_indices_out_capa5 = %s" % path_indices_out_capa5)
print("path_modelo_reductor_features = %s" % path_modelo_reductor_features)
print("balancear en C5 (en C6 también hay otro) = " + str(balancear))

######################## FUNCIONES #######################################################

with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore",
                            message="Bins whose width are too small.*")  # Ignorar los warnings del tramificador (KBinsDiscretizer)


np.random.seed(12345)

print("\n" + (datetime.datetime.now()).strftime(
    "%Y%m%d_%H%M%S") + " **** CAPA 6 - Crear almacenar y evaluar varios modelos (para cada subgrupo) ****")
print("Tipo de problema: CLASIFICACION DICOTOMICA (target es boolean)")

print("PARAMETROS: ")
pathFeaturesSeleccionadas = dir_subgrupo + "FEATURES_SELECCIONADAS.csv"
modoDebug = False  # En modo debug se pintan los dibujos. En otro caso, se evita calculo innecesario
umbralCasosSuficientesClasePositiva = 50
granProbTargetUno = 50  # De todos los target=1, nos quedaremos con los granProbTargetUno (en tanto por cien) MAS probables. Un valor de 100 o mayor anula este parámetro
balancearConSmoteSoloTrain = True
umbralFeaturesCorrelacionadas = 0.96  # Umbral aplicado para descartar features cuya correlacion sea mayor que él
umbralNecesarioCompensarDesbalanceo = 1  # Umbral de desbalanceo clase positiva/negativa. Si se supera, es necesario hacer oversampling de minoritaria (SMOTE) o undersampling de mayoritaria (borrar filas)
cv_todos = 10  # CROSS_VALIDATION: número de iteraciones. Sirve para evitar el overfitting
fraccion_train = 0.50  # Fracción de datos usada para entrenar
fraccion_test = 0.25  # Fracción de datos usada para testear (no es validación)
fraccion_valid = 1.00 - (fraccion_train + fraccion_test)

######### ID de subgrupo #######
partes = dir_subgrupo.split("/")
id_subgrupo = ""
for parte in partes:
    if (parte != ''):
        id_subgrupo = parte

########### Rutas #########
pathCsvCompleto = dir_subgrupo + "COMPLETO.csv"
pathCsvReducido = dir_subgrupo + "REDUCIDO.csv"
pathCsvPredichos = dir_subgrupo + "TARGETS_PREDICHOS.csv"
pathCsvPredichosIndices = dir_subgrupo + "TARGETS_PREDICHOS.csv_indices"
pathCsvFinalFuturo = dir_subgrupo + desplazamientoAntiguedad + "_" + id_subgrupo + "_COMPLETO_PREDICCION.csv"
dir_subgrupo_img = dir_subgrupo + "img/"
umbralProbTargetTrue = float("0.50")

print("dir_subgrupo: %s" % dir_subgrupo)
print("modoTiempo: %s" % modoTiempo)
print("desplazamientoAntiguedad: %s" % desplazamientoAntiguedad)
print("pathCsvReducido: %s" % pathCsvReducido)
print("dir_subgrupo_img = %s" % dir_subgrupo_img)
print("umbralProbTargetTrue = " + str(umbralProbTargetTrue))
print("balancearConSmoteSoloTrain = " + str(balancearConSmoteSoloTrain))
print("umbralFeaturesCorrelacionadas = " + str(umbralFeaturesCorrelacionadas))

################## MAIN ###########################################################

if pathCsvCompleto.endswith('.csv') and os.path.isfile(pathCsvCompleto) and os.stat(pathCsvCompleto).st_size > 0:

    print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " ----- leerFeaturesyTarget ------")
    print("PARAMS --> " + pathCsvCompleto + "|" + dir_subgrupo_img + "|" + str(
        compatibleParaMuchasEmpresas) + "|" + pathModeloOutliers + "|" + modoTiempo + "|" + str(modoDebug)
          + "|" + str(maxFilasEntrada))

    print("Cargar datos (CSV)...")
    df = pd.read_csv(filepath_or_buffer=pathCsvCompleto, sep='|')
    print("df (LEIDO): " + str(df.shape[0]) + " x " + str(
        df.shape[1]))
    df.sort_index(
        inplace=True)  # Reordenando segun el indice (para facilitar el testIntegracion)
    df.to_csv(pathCsvIntermedio + ".entrada", index=True,
                                  sep='|')  # NO BORRAR: UTIL para testIntegracion

    if int(maxFilasEntrada) < df.shape[0]:
        print("df (APLICANDO MAXIMO): " + str(df.shape[0]) + " x " + str(
            df.shape[1]))
        df = df.sample(int(maxFilasEntrada), replace=False)

    print('INICIO REDORDEN ÍNDICES...')
    df.sort_index(
        inplace=True)  # Reordenando segun el indice (para facilitar el testIntegracion)
    print('FIN REDORDEN ÍNDICES...')

    ################ Eliminación de columnas no numéricas
    print('INICIO ELIMINACIÓN COLUMNAS NO NUMÉRICAS...')
    df = df.drop('empresa', axis=1)
    df = df.drop('mercado', axis=1)
    print('FIN ELIMINACIÓN COLUMNAS NO NUMÉRICAS...')

    ######################### Eliminación de filas con algún nulo (para el futuro, el target es excepción
    # porque siempre es nulo)
    if modoTiempo == "pasado":
        df = df.dropna(how='any', axis=0)
        #  Features (X) y Targets (Y)
        y = (df[['TARGET']] == 1)  # Convierto de int a boolean
        x = df.drop(['TARGET'], axis=1)
    elif modoTiempo == "futuro":
        df = df.drop('TARGET', axis=1).dropna(how='any', axis=0)
        #  Features (X) y Targets (Y)
        x = df

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

######################### GENERACIÓN DE MODELO (PASADO) ###################################
if modoTiempo == "pasado":

    # Se recodifica el target a 0/1
    df['TARGET'] = df['TARGET'].astype(np.int64)

    # Se separa features y target
    x = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]

    ################## Splitting train, test, validation
    from sklearn.model_selection import train_test_split
    print('INICIO SPLIT...')
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3, stratify=y, random_state=0)
    print('FIN SPLIT...')

    ############### you should not fit any preprocessing algorithm (PCA, StandardScaler...)
    # on the whole dataset, but only on the training set,
    ############### you should do most pre-processing steps (encoding, normalization/standardization, etc)
    # before under/over-sampling the data.

    ################## Escalado
    import numpy as np
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from pickle import dump

    # define scaler
    scaler = RobustScaler()
    # fit scaler on the training dataset
    print('INICIO ESCALADO...')
    # fit and transform the training dataset
    x_train = scaler.fit_transform(x_train)
    # Guardado de scaler
    dump(scaler, open('scaler.pkl', 'wb'))

    print('FIN ESCALADO...')

    # ################# INICIO DE SMOTE
    # # JUSTO ANTES DE fitear el modelo, se aplica SMOTE sólo en el train
    # print("INICIO SMOTE EN TRAIN...")
    # resample = SMOTETomek()
    # df_mayoritaria = y_train[y_train['TARGET'] == False]
    # df_minoritaria = y_train[y_train['TARGET'] == True]
    # print("SMOTE antes (mayoritaria + minoritaria): %d" % x_train.shape[0])
    # print("df_mayoritaria:" + str(len(df_mayoritaria)))
    # print("df_minoritaria:" + str(len(df_minoritaria)))
    # x_train, y_train = resample.fit_sample(x_train, y_train)
    # df_mayoritaria = y_train[y_train['TARGET']  == False]
    # df_minoritaria = y_train[y_train['TARGET']  == True]
    # print("SMOTE después (mayoritaria + minoritaria): %d" % x_train.shape[0])
    # print("df_mayoritaria:" + str(len(df_mayoritaria)))
    # print("df_minoritaria:" + str(len(df_minoritaria)))
    # ################# FIN DE SMOTE

    # Hiperparámetros del modelo
    EPOCHS = 50
    BATCH_SIZE = 64
    LEARNING_RATE = 0.1


    ## train data
    class trainData(Dataset):
        def __init__(self, X_data, y_data):
            self.X_data = X_data
            self.y_data = y_data

        def __getitem__(self, index):
            return self.X_data[index], self.y_data[index]

        def __len__(self):
            return len(self.X_data)


    train_data = trainData(torch.FloatTensor(x_train),
                           torch.FloatTensor(y_train))


    ## test data
    class testData(Dataset):
        def __init__(self, X_data):
            self.X_data = X_data

        def __getitem__(self, index):
            return self.X_data[index]

        def __len__(self):
            return len(self.X_data)


    test_data = testData(torch.FloatTensor(x_test))

    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1)

    # El número de neuronas de entrada sería el número de features
    neuronasEntrada=x_train.columns

    # Define NN
    class binaryClassification(nn.Module):
        def __init__(self):
            super(binaryClassification, self).__init__()
            self.layer_1 = nn.Linear(neuronasEntrada, 64)
            self.layer_2 = nn.Linear(64, 64)
            self.layer_out = nn.Linear(64, 1)

            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=0.1)
            self.batchnorm1 = nn.BatchNorm1d(64)
            self.batchnorm2 = nn.BatchNorm1d(64)

        def forward(self, inputs):
            x = self.relu(self.layer_1(inputs))
            x = self.batchnorm1(x)
            x = self.relu(self.layer_2(x))
            x = self.batchnorm2(x)
            x = self.dropout(x)
            x = self.layer_out(x)

            return x




    model = binaryClassification()
    model.to(device)
    print(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


    # Train
    def binary_acc(y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum / y_test.shape[0]
        acc = torch.round(acc * 100)

        return acc


    model.train()


    ############ GUARDADO DE MODELO ###############################
    # save the model
    dump(model, open('model.pkl', 'wb'))
    ##################################################################

    # Se calculan loss y accuracy
    for e in range(1, EPOCHS + 1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        print(
            f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')

    # Test
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

    # Confusion matrix
    confusion_matrix(y_test, y_pred_list)

    # Classification report
    print(classification_report(y_test, y_pred_list))




######################### USO DE MODELO (FUTURO) ###################################
elif modoTiempo == "futuro":

    test_loader = DataLoader(dataset=x, batch_size=1)

    #  Features (X) y Targets (Y)
    from sklearn.model_selection import train_test_split

    # load model to make predictions on new data
    from pickle import load
    # load the model
    print('INICIO CARGA DE MODELO...')
    model = load(open('model.pkl', 'rb'))
    print('FIN CARGA DE MODELO...')

    # make predictions
    print('INICIO PREDICCIÓN...')
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in x:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    print('FIN PREDICCIÓN...')



