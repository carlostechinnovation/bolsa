import sys
import os
import pandas as pd
from pathlib import Path
from random import sample, choice

from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from pandas import DataFrame
from sklearn.covariance import EllipticEnvelope
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer, KBinsDiscretizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.feature_selection import RFECV, RFE
from sklearn import metrics
import numpy as np
from sklearn import linear_model
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
import pickle
from sklearn.impute import SimpleImputer
import warnings
import datetime
from sklearn.pipeline import make_pipeline
from sklearn.manifold import TSNE
import math
import sys
import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours, NearMiss, NeighbourhoodCleaningRule
from numpy import mean
from scipy.stats import stats, chi2
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.fixes import loguniform
from pathlib import Path
from sklearn import svm
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, recall_score, make_scorer, \
    precision_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score, train_test_split, \
    ParameterGrid
from sklearn.calibration import CalibratedClassifierCV
from shutil import copyfile
import os.path
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.tree import export_graphviz
from subprocess import call

from statsmodels.tools.eval_measures import rmse
from xgboost import XGBClassifier
from matplotlib import pyplot
import datetime
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import time
import matplotlib.patches as mpatches


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

######################### GENERACIÓN DE MODELO (PASADO) ###################################
if modoTiempo == "pasado":

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



    ###################### MODELO LOGISTIC REGRESSION ########################
    model = LogisticRegression(solver='lbfgs')
    model.fit(x_train, y_train)

    ###################### FIN MODELO LOGISTIC REGRESSION ###################

    ############ GUARDADO DE MODELO ###############################33
    # save the model
    dump(model, open('model.pkl', 'wb'))

    ##################### ANÁLISIS DE TRAIN
    # Precisión de train
    print('INICIO PRECISIÓN DE TRAIN...')
    ypred_train = model.predict(x_train)
    precision_train = precision_score(y_train, ypred_train)
    print('TRAIN Precision:', precision_train)
    print('FIN PRECISIÓN DE TRAIN...')

    ##################### ANÁLISIS DE TEST
    # Precisión de test
    print('INICIO PRECISIÓN DE TEST...')
    ypred = model.predict(x_test)
    precision = precision_score(y_test, ypred)
    print('Test Precision:', precision)
    print('FIN PRECISIÓN DE TEST...')

    ###################### PRINT TRAIN VS TEST
    from sklearn import datasets, linear_model
    from sklearn.model_selection import cross_val_predict
    # define lists to collect scores
    train_scores, test_scores = list(), list()
    # define the tree depths to evaluate
    values = [i for i in range(1, 21)]
    # evaluate a decision tree for each depth
    for i in values:
        # configure the model
        model = DecisionTreeClassifier(max_depth=i)
        # CV
        y_pred = cross_val_predict(model, x_train, y_train, cv=5)
        # fit model on the training dataset
        model.fit(x_train, y_train)
        # evaluate on the train dataset
        train_yhat = model.predict(x_train)
        train_prec = precision_score(y_train, train_yhat)
        train_scores.append(train_prec)
        # evaluate on the test dataset
        test_yhat = model.predict(x_test)
        test_prec = precision_score(y_test, test_yhat)
        test_scores.append(test_prec)
        # summarize progress
        print('>%d, train: %.3f, test: %.3f' % (i, train_prec, test_prec))
    # plot of train and test scores vs tree depth
    pyplot.plot(values, train_scores, '-o', label='Train')
    pyplot.plot(values, test_scores, '-o', label='Test')
    pyplot.legend()
    pyplot.show()

######################### USO DE MODELO (FUTURO) ###################################
elif modoTiempo == "futuro":

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
    ypred = model.predict(x)
    print('FIN PREDICCIÓN...')















