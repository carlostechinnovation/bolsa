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
from sklearn.model_selection import StratifiedKFold
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


################# FUNCIONES ########################################



################## MAIN ###########################################################

if pathCsvCompleto.endswith('.csv') and os.path.isfile(pathCsvCompleto) and os.stat(pathCsvCompleto).st_size > 0:

    print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " ----- leerFeaturesyTarget ------")
    print("PARAMS --> " + pathCsvCompleto + "|" + dir_subgrupo_img + "|" + str(
        compatibleParaMuchasEmpresas) + "|" + pathModeloOutliers + "|" + modoTiempo + "|" + str(modoDebug)
          + "|" + str(maxFilasEntrada))

    print("Cargar datos (CSV)...")
    entradaFeaturesYTarget = pd.read_csv(filepath_or_buffer=pathCsvCompleto, sep='|')
    print("entradaFeaturesYTarget (LEIDO): " + str(entradaFeaturesYTarget.shape[0]) + " x " + str(
        entradaFeaturesYTarget.shape[1]))
    entradaFeaturesYTarget.sort_index(
        inplace=True)  # Reordenando segun el indice (para facilitar el testIntegracion)
    entradaFeaturesYTarget.to_csv(pathCsvIntermedio + ".entrada", index=True,
                                  sep='|')  # NO BORRAR: UTIL para testIntegracion

    if int(maxFilasEntrada) < entradaFeaturesYTarget.shape[0]:
        print("entradaFeaturesYTarget (APLICANDO MAXIMO): " + str(entradaFeaturesYTarget.shape[0]) + " x " + str(
            entradaFeaturesYTarget.shape[1]))
        entradaFeaturesYTarget = entradaFeaturesYTarget.sample(int(maxFilasEntrada), replace=False)

    entradaFeaturesYTarget.sort_index(
        inplace=True)  # Reordenando segun el indice (para facilitar el testIntegracion)
    entradaFeaturesYTarget.to_csv(pathCsvIntermedio + ".entrada_tras_maximo.csv", index=True,
                                  sep='|')  # NO BORRAR: UTIL para testIntegracion
    entradaFeaturesYTarget.to_csv(pathCsvIntermedio + ".entrada_tras_maximo_INDICES.csv",
                                  columns=[])  # NO BORRAR: UTIL para testIntegracion

    num_nulos_por_fila_1 = np.logical_not(entradaFeaturesYTarget.isnull()).sum()
    indiceFilasFuturasTransformadas1 = entradaFeaturesYTarget.index.values  # DEFAULT
    indiceFilasFuturasTransformadas2 = entradaFeaturesYTarget.index.values  # DEFAULT

    ################# Borrado de columnas nulas enteras ##########
    print("MISSING VALUES (COLUMNAS) - Borramos las columnas (features) que sean siempre NaN...")
    entradaFeaturesYTarget2 = entradaFeaturesYTarget.dropna(axis=1,
                                                            how='all')  # Borrar COLUMNA si TODOS sus valores tienen NaN
    print(
        "entradaFeaturesYTarget2 (columnas nulas borradas): " + str(entradaFeaturesYTarget2.shape[0]) + " x " + str(
            entradaFeaturesYTarget2.shape[1]))
    # entradaFeaturesYTarget2.to_csv(path_csv_completo + "_TEMP01", index=True, sep='|')  # UTIL para testIntegracion

    ################# Borrado de filas que tengan algun hueco (tratamiento espacial para el futuro con sus columnas: TARGET y otras) #####
    print("MISSING VALUES (FILAS)...")

    if modoTiempo == "futuro":
        print("Nos quedamos solo con las velas con antiguedad=0 (futuras)...")
        entradaFeaturesYTarget2 = entradaFeaturesYTarget2[entradaFeaturesYTarget2.antiguedad == 0]
        print("entradaFeaturesYTarget2:" + str(entradaFeaturesYTarget2.shape[0]) + " x " + str(
            entradaFeaturesYTarget2.shape[1]))
        # print(entradaFeaturesYTarget2.head())

    print("Porcentaje de MISSING VALUES en cada columna del dataframe de entrada (mostramos las que superen 20%):")
    missing = pd.DataFrame(entradaFeaturesYTarget2.isnull().sum()).rename(columns={0: 'total'})
    missing['percent'] = missing['total'] / len(entradaFeaturesYTarget2)  # Create a percentage missing
    missing_df = missing.sort_values('percent', ascending=False)
    missing_df = missing_df[missing_df['percent'] > 0.20]
    print(missing_df.to_string())  # .drop('TARGET')

    # print("Pasado o Futuro: Transformacion en la que borro filas. Por tanto, guardo el indice...")
    indiceFilasFuturasTransformadas1 = entradaFeaturesYTarget2.index.values

    print(
        "Borrar columnas especiales (idenficadoras de fila): empresa | antiguedad | mercado | anio | mes | dia | hora | minuto...")
    entradaFeaturesYTarget2 = entradaFeaturesYTarget2.drop('empresa', axis=1).drop('antiguedad', axis=1).drop(
        'mercado',
        axis=1).drop(
        'anio', axis=1).drop('mes', axis=1).drop('dia', axis=1).drop('hora', axis=1).drop('minuto', axis=1)

    print("Borrar columnas dinamicas que no aportan nada: volumen | high | low | close | open ...")
    entradaFeaturesYTarget2 = entradaFeaturesYTarget2.drop('volumen', axis=1).drop('high', axis=1).drop('low',
                                                                                                        axis=1).drop(
        'close', axis=1).drop('open', axis=1)

    # entradaFeaturesYTarget2.to_csv(path_csv_completo + "_TEMP02", index=True, sep='|')  # UTIL para testIntegracion

    print("entradaFeaturesYTarget2: " + str(entradaFeaturesYTarget2.shape[0]) + " x " + str(
        entradaFeaturesYTarget2.shape[1]))
    # print(entradaFeaturesYTarget2.head())

    if modoTiempo == "pasado":
        print(
            "MISSING VALUES (FILAS) - Borramos las FILAS que tengan 1 o mas valores NaN porque son huecos que no deberían estar...")
        entradaFeaturesYTarget3 = entradaFeaturesYTarget2.dropna(axis=0,
                                                                 how='any')  # Borrar FILA si ALGUNO sus valores tienen NaN

    elif modoTiempo == "futuro":
        print(
            "MISSING VALUES (FILAS) - Para el FUTURO; el target es NULO siempre, pero borramos las filas que tengan ademas otros NULOS...")
        entradaFeaturesYTarget3 = entradaFeaturesYTarget2.drop('TARGET', axis=1).dropna(axis=0,
                                                                                        how='any')  # Borrar FILA si ALGUNO sus valores tienen NaN

    print("Pasado o futuro: Transformacion en la que he borrado filas. Por tanto, guardo el indice...")
    indiceFilasFuturasTransformadas2 = entradaFeaturesYTarget3.index.values
    print("indiceFilasFuturasTransformadas2: " + str(indiceFilasFuturasTransformadas2.shape[0]))
    # print(indiceFilasFuturasTransformadas2)

    print("entradaFeaturesYTarget3 (filas con algun nulo borradas):" + str(
        entradaFeaturesYTarget3.shape[0]) + " x " + str(entradaFeaturesYTarget3.shape[1]))
    # entradaFeaturesYTarget3.to_csv(path_csv_completo + "_TEMP03", index=True, sep='|')  # UTIL para testIntegracion

    # Limpiar OUTLIERS
    # URL: https://scikit-learn.org/stable/modules/outlier_detection.html
    if modoTiempo == "pasado":
        detector_outliers = IsolationForest()
        df3aux = entradaFeaturesYTarget3.drop('TARGET', axis=1)
        detector_outliers.fit(df3aux)  # fit 10 trees
        pickle.dump(detector_outliers, open(pathModeloOutliers, 'wb'))
    else:
        df3aux = entradaFeaturesYTarget3  # FUTURO

    # Pasado y futuro:
    print("Cargando modelo detector de outliers: " + pathModeloOutliers)
    detector_outliers = pickle.load(open(pathModeloOutliers, 'rb'))
    flagAnomaliasDf = pd.DataFrame(
        {'marca_anomalia': detector_outliers.predict(df3aux)})  # vale -1 es un outlier; si es un 1, no lo es

    indice3 = entradaFeaturesYTarget3.index  # lo guardo para pegarlo luego
    entradaFeaturesYTarget3.reset_index(drop=True, inplace=True)
    flagAnomaliasDf.reset_index(drop=True, inplace=True)
    entradaFeaturesYTarget4 = pd.concat([entradaFeaturesYTarget3, flagAnomaliasDf],
                                        axis=1)  # Column Bind, manteniendo el índice del DF izquierdo
    entradaFeaturesYTarget4.set_index(indice3, inplace=True)  # ponemos el indice que tenia el DF de la izquierda

    entradaFeaturesYTarget4 = entradaFeaturesYTarget4.loc[
        entradaFeaturesYTarget4['marca_anomalia'] == 1]  # Cogemos solo las que no son anomalias
    entradaFeaturesYTarget4 = entradaFeaturesYTarget4.drop('marca_anomalia', axis=1)  # Quitamos la columna auxiliar

    print("entradaFeaturesYTarget4 (sin outliers):" + str(entradaFeaturesYTarget4.shape[0]) + " x " + str(
        entradaFeaturesYTarget4.shape[1]))
    # entradaFeaturesYTarget4.to_csv(path_csv_completo + "_TEMP04", index=True, sep='|')  # UTIL para testIntegracion

    # ENTRADA: features (+ target)

    # Si hay POCAS empresas
    if compatibleParaMuchasEmpresas is False or modoTiempo == "futuro":
        if modoTiempo == "pasado":
            featuresFichero = entradaFeaturesYTarget4.drop('TARGET', axis=1)
            targetsFichero = (entradaFeaturesYTarget4[['TARGET']] == 1)  # Convierto de int a boolean
        elif modoTiempo == "futuro":
            featuresFichero = entradaFeaturesYTarget4
            targetsFichero = pd.DataFrame({'TARGET': []})  # DEFAULT: Vacio (caso futuro)


    # SOLO PARA EL PASADO Si hay MUCHAS empresas (UNDER-SAMPLING para reducir los datos -útil para miles de empresas, pero puede quedar sobreentrenado, si borro casi todas las minoritarias-)
    else:

        if (balancear == True):
            print("BALANCEAR los casos positivos y negativos, haciendo downsampling de la clase mayoritaria...")
            print("URL: https://elitedatascience.com/imbalanced-classes")
            ift_minoritaria = entradaFeaturesYTarget4[entradaFeaturesYTarget4.TARGET == True]
            print("ift_minoritaria (original):" + str(ift_minoritaria.shape[0]) + " x " + str(
                ift_minoritaria.shape[1]))
            num_filas_azar = 5 * ift_minoritaria.shape[0]
            print("num_filas_azar:" + str(num_filas_azar))
            ift_mayoritaria = entradaFeaturesYTarget4.loc[
                np.random.choice(entradaFeaturesYTarget4.index, num_filas_azar)]
            ift_mayoritaria = ift_mayoritaria[ift_mayoritaria.TARGET == False]
            print(
                "ift_mayoritaria (se han borrado filas, pero no muchas):" + str(
                    ift_mayoritaria.shape[0]) + " x " + str(
                    ift_mayoritaria.shape[1]))
            print("ift_minoritaria (con oversampling): " + str(ift_minoritaria.shape[0]) + " x " + str(
                ift_minoritaria.shape[1]))
            print("Tasa de desbalanceo entre clases = " + str(ift_mayoritaria.shape[0]) + "/" + str(
                ift_minoritaria.shape[0]) + " = " + str(ift_mayoritaria.shape[0] / ift_minoritaria.shape[0]))
            # Juntar ambas clases ya BALANCEADAS. Primero vacío el dataset
            entradaFeaturesYTarget5 = ift_mayoritaria.append(ift_minoritaria)
            print("Las clases ya están balanceadas:")
            print("ift_balanceadas:" + str(entradaFeaturesYTarget5.shape[0]) + " x " + str(
                entradaFeaturesYTarget5.shape[1]))

        else:
            print("NO balanceamos clases en capa 5 (pero seguramente sí en capa 6 solo sobre dataset de TRAIN)!!!")
            ift_minoritaria = entradaFeaturesYTarget4[entradaFeaturesYTarget4.TARGET == True]
            ift_mayoritaria = entradaFeaturesYTarget4[entradaFeaturesYTarget4.TARGET == False]
            print("Tasa de desbalanceo entre clases = " + str(ift_mayoritaria.shape[0]) + "/" + str(
                ift_minoritaria.shape[0]) + " = " + str(ift_mayoritaria.shape[0] / ift_minoritaria.shape[0]))
            entradaFeaturesYTarget5 = entradaFeaturesYTarget4

        # entradaFeaturesYTarget5.to_csv(path_csv_completo + "_TEMP05", index=True, sep='|')  # UTIL para testIntegracion
        featuresFichero = entradaFeaturesYTarget5.drop('TARGET', axis=1)
        targetsFichero = entradaFeaturesYTarget5[['TARGET']]
        targetsFichero = (targetsFichero[['TARGET']] == 1)  # Convierto de int a boolean

    ##################################################

    # print("FEATURES (sample):")
    # print(featuresFichero.head())
    # print("TARGETS (sample):")
    # print(targetsFichero.head())

    if modoDebug and modoTiempo == "pasado":
        print("FUNCIONES DE DENSIDAD (sin nulos, pero antes de normalizar):")
        for column in featuresFichero:
            path_dibujo = dir_subgrupo_img + column + ".png"
            print("Guardando distrib de col: " + column + " en fichero: " + path_dibujo)
            datos_columna = featuresFichero[column]
            sns.distplot(datos_columna, kde=False, color='red', bins=dibujoBins)
            plt.title(column, fontsize=10)
            plt.savefig(path_dibujo, bbox_inches='tight')
            plt.clf();
            plt.cla();
            plt.close()  # Limpiando dibujo

    # NORMALIZAR, PERO SIN TRAMIFICAR: leer apartado 4.3 de https://eprints.ucm.es/56355/1/TFM_MPP_Jul19%20%281%29Palau.pdf

    print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " ----- normalizarFeatures ------")
    print(
        "NORMALIZACION: hacemos que todas las features tengan distribución gaussiana media 0 y varianza 1. El target no se toca.")
    print("PARAMS --> " + path_modelo_normalizador + "|" + modoTiempo + "|" + str(modoDebug))
    print("featuresFichero: " + str(featuresFichero.shape[0]) + " x " + str(featuresFichero.shape[1]))
    print("path_modelo_normalizador: " + path_modelo_normalizador)
    print("pathCsvIntermedio: " + pathCsvIntermedio)

    ################################### NORMALIZACIÓN YEO-JOHNSON ####################################################
    print("Normalizando cada feature...")
    # Vamos a normalizar z-score (media 0, std_dvt=1), pero yeo-johnson tiene un bug (https://github.com/scipy/scipy/issues/10821) que se soluciona sumando una constante a toda la matriz, lo cual no afecta a la matriz normalizada
    featuresFichero = featuresFichero + 1.015815

    if modoTiempo == "pasado":
        # Con el "normalizador COMPLEJO" solucuionamos este bug: https://github.com/scikit-learn/scikit-learn/issues/14959  --> Aplicar los cambios indicados a:_/home/carloslinux/Desktop/PROGRAMAS/anaconda3/envs/BolsaPython/lib/python3.7/site-packages/sklearn/preprocessing/_data.py

        
        modelo_normalizador = make_pipeline(StandardScaler(with_std=False),
                                            PowerTransformer(method='yeo-johnson', standardize=True,
                                                             copy=True), ).fit(
            featuresFichero)  # COMPLEJO
        # modelo_normalizador = PowerTransformer(method='yeo-johnson', standardize=True, copy=True).fit(featuresFichero)
        pickle.dump(modelo_normalizador, open(path_modelo_normalizador, 'wb'))

    # Pasado o futuro: Cargar normalizador
    modelo_normalizador = pickle.load(open(path_modelo_normalizador, 'rb'))

    print("Aplicando normalizacion, manteniendo indices y nombres de columnas...")
    featuresFicheroNorm = pd.DataFrame(data=modelo_normalizador.transform(featuresFichero),
                                       index=featuresFichero.index,
                                       columns=featuresFichero.columns)

    print("featuresFicheroNorm:" + str(featuresFicheroNorm.shape[0]) + " x " + str(featuresFicheroNorm.shape[1]))
    featuresFicheroNorm.to_csv(pathCsvIntermedio + ".normalizado.csv", index=True,
                               sep='|')  # UTIL para testIntegracion

    if modoDebug and modoTiempo == "pasado":
        print("FUNCIONES DE DENSIDAD (normalizadas):")
        for column in featuresFicheroNorm:
            path_dibujo = dir_subgrupo_img + column + "_NORM.png"
            print("Guardando distrib de col normalizada: " + column + " en fichero: " + path_dibujo)
            datos_columna = featuresFicheroNorm[column]
            sns.distplot(datos_columna, kde=False, color='red', bins=dibujoBins)
            plt.title(column + " (NORM)", fontsize=10)
            plt.savefig(path_dibujo, bbox_inches='tight')
            plt.clf();
            plt.cla();
            plt.close()  # Limpiando dibujo

    featuresFichero3 = featuresFicheroNorm

    # NO NORMALIZAR y NO TRAMIFICAR
    # featuresFichero2 = featuresFichero1
    # featuresFichero3 = featuresFichero2

    # -----  Comprobar las clases del target:
    print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " ----- comprobarSuficientesClasesTarget ------")
    print("featuresFicheroNorm: " + str(featuresFichero3.shape[0]) + " x " + str(
        featuresFichero3.shape[1]) + "  Y  " + "targetsFichero: " + str(targetsFichero.shape[0]) + " x " + str(
        targetsFichero.shape[1]))

    y_unicos = np.unique(targetsFichero)
    # print("Clases encontradas en el target: ")
    # print(y_unicos)
    numclases = y_unicos.size


    if (modoTiempo == "pasado" and numclases <= 1):
        print("El subgrupo solo tiene " + str(numclases) + " clases en el target. Abortamos...")
    else:

        print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " ----- reducirFeaturesYGuardar ------")
        print("path_modelo_reductor_features --> " + path_modelo_reductor_features)
        print("path_modelo_pca --> " + path_modelo_pca)
        print("path_pesos_pca --> " + path_pesos_pca)
        print("featuresFichero3: " + str(featuresFichero3.shape[0]) + " x " + str(featuresFichero3.shape[1]))
        print("targetsFichero: " + str(targetsFichero.shape[0]) + " x " + str(targetsFichero.shape[1]))
        print("pathCsvReducido --> " + pathCsvReducido)
        print("pathCsvFeaturesElegidas --> " + pathCsvFeaturesElegidas)
        print("varianza (PCA) --> " + str(varianza))
        print("dir_subgrupo_img --> " + dir_subgrupo_img)
        print("modoTiempo: " + modoTiempo)
        print("maxFeatReducidas: " + maxFeatReducidas)

        print("**** REDUCCION DE DIMENSIONES*****")

        print("** Recursive Feature Elimination (RFE) (que se parece a la técnica Step-wise) **")
        # Create the RFE object and compute a cross-validated score.

        # Comparación de clasificadores
        print('CLASIFICADORES - DENTRO DEL HILO DE EJECUCIÓN')

        if modoTiempo == "pasado":

            estimador_interno = AdaBoostClassifier(n_estimators=50, learning_rate=0.3)

            # accuracy,balanced_accuracy,average_precision,neg_brier_score,f1,f1_micro,f1_macro,f1_weighted,roc_auc,roc_auc_ovr,roc_auc_ovo,roc_auc_ovr_weighted,roc_auc_ovo_weighted
            # Es mejor roc_auc que f1 y que average_precision/precision. El roc_auc_ovo_weighted no mejora, y roc_auc_ovr_weighted es peor.
            rfecv_scoring = 'roc_auc'

            # The "accuracy" scoring is proportional to the number of correct classifications
            num_filas_en_cada_trozo = targetsFichero.shape[0] / cv_todos
            if num_filas_en_cada_trozo < 10:  # La funcion fit() de RFECV exige que haya al menos 10 muestras en el vector target
                n_splits_corregido = math.floor((cv_todos / 10) * num_filas_en_cada_trozo)
            else:
                n_splits_corregido = cv_todos

            print("n_splits_corregido -->" + str(n_splits_corregido))
            rfecv_modelo = RFECV(estimator=estimador_interno, step=rfecv_step, min_features_to_select=4,
                                 cv=StratifiedKFold(n_splits=n_splits_corregido, shuffle=True), scoring=rfecv_scoring,
                                 verbose=0, n_jobs=-1)
            print("rfecv_modelo -> fit ...")
            targetsLista = targetsFichero["TARGET"].tolist()
            rfecv_modelo.fit(featuresFichero3, targetsLista)
            print("rfecv_modelo -> dump ...")
            pickle.dump(rfecv_modelo, open(path_modelo_reductor_features, 'wb'))

        ################ REDUCCION DE FEATURES para pasado y futuro: ###########
        rfecv_modelo = pickle.load(open(path_modelo_reductor_features, 'rb'))
        print("Numero original de features: %d" % featuresFichero3.shape[1])
        print("Numero optimo de features: %d" % rfecv_modelo.n_features_)

        if rfecv_modelo.n_features_ > int(maxFeatReducidas):
            print(
                "El reductor de dimensiones no es capaz de reducir a un numero razonable/manejable de dimensiones (pocas). Por tanto, no seguimos calculando nada para este subgrupo.")

        else:

            # Plot number of features VS. cross-validation scores
            if modoDebug and modoTiempo == "pasado":
                path_dibujo_rfecv = dir_subgrupo_img + "SELECCION_VARIABLES_RFECV" ".png"
                plt.figure()
                plt.xlabel("Number of features selected")
                plt.ylabel("Cross validation score (nb of correct classifications)")
                plt.plot(range(1, len(rfecv_modelo.grid_scores_) + 1), rfecv_modelo.grid_scores_)
                plt.title("RFECV", fontsize=10)
                plt.savefig(path_dibujo_rfecv, bbox_inches='tight')
                plt.clf();
                plt.cla();
                plt.close()  # Limpiando dibujo

            columnas = featuresFichero3.columns
            numColumnas = columnas.shape[0]
            columnasSeleccionadas = []
            for i in range(numColumnas):
                if (rfecv_modelo.support_[i] == True):
                    columnasSeleccionadas.append(columnas[i])

            # print("Mascara de features seleccionadas (rfecv_modelo.support_):")
            # print(rfecv_modelo.support_)
            # print("El ranking de importancia de las features (rfecv_modelo.ranking_) no distingue las features mas importantes dentro de las seleccionadas:")
            # print(rfecv_modelo.ranking_)

            featuresFichero3Elegidas = featuresFichero3[columnasSeleccionadas]
            print("Features seleccionadas escritas en: " + pathCsvFeaturesElegidas)
            featuresFichero3Elegidas.head(1).to_csv(pathCsvFeaturesElegidas, index=False, sep='|')

            ########### PCA: base de funciones ortogonales (con combinaciones de features) ########
            if True:
                print("** PCA (Principal Components Algorithm) **")

                if modoTiempo == "pasado":
                    print(
                        "Usando PCA, creamos una NUEVA BASE DE FEATURES ORTOGONALES y cogemos las que tengan un impacto agregado sobre el X% de la varianza del target. Descartamos el resto.")
                    # modelo_pca_subgrupo = PCA(n_components=varianza, svd_solver='full')  # Variaza acumulada sobre el target
                    modelo_pca_subgrupo = PCA(n_components='mle',
                                              svd_solver='full')  # Metodo "MLE de Minka": https://vismod.media.mit.edu/tech-reports/TR-514.pdf
                    # modelo_pca_subgrupo = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0,
                    #                            n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07,
                    #                            metric='euclidean', init='random', verbose=0, random_state=None,
                    #                            method='barnes_hut', angle=0.5,
                    #                            n_jobs=-1)  # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
                    print(modelo_pca_subgrupo)
                    featuresFichero3_pca = modelo_pca_subgrupo.fit_transform(featuresFichero3Elegidas)
                    print("modelo_pca_subgrupo -> dump ...")
                    pickle.dump(modelo_pca_subgrupo, open(path_modelo_pca, 'wb'))
                else:
                    print("modelo_pca_subgrupo -> load ...")
                    modelo_pca_subgrupo = pickle.load(open(path_modelo_pca, 'rb'))
                    print(modelo_pca_subgrupo)
                    featuresFichero3_pca = modelo_pca_subgrupo.transform(featuresFichero3Elegidas)

                print("Dimensiones del dataframe tras PCA: " + str(featuresFichero3_pca.shape[0]) + " x " + str(
                    featuresFichero3_pca.shape[1]))

                print(
                    "Las features están ya normalizadas, reducidas y en base ortogonal PCA. DESCRIBIMOS lo que hemos hecho y GUARDAMOS el dataset.")
                num_columnas_pca = featuresFichero3_pca.shape[1]
                columnas_pca = ["pca_" + f"{i:0>2}" for i in
                                range(num_columnas_pca)]  # Hacemos left padding con la funcion f-strings
                featuresFichero3_pca_df = DataFrame(featuresFichero3_pca, columns=columnas_pca,
                                                       index=featuresFichero3.index)
                print(featuresFichero3_pca_df.head())
                featuresFichero3Elegidas = featuresFichero3_pca_df

                print("Matriz de pesos de las features en la base de funciones PCA: " + path_pesos_pca)
                pcaMatriz = pd.DataFrame(modelo_pca_subgrupo.components_)
                pcaMatriz.columns = columnasSeleccionadas
                columnas_pca_df = pd.DataFrame(columnas_pca)
                pcaMatriz = pd.concat([columnas_pca_df, pcaMatriz], axis=1)
                pcaMatriz.to_csv(path_pesos_pca, index=False, sep='|')

            ### Guardar a fichero
            # print("Muestro las features + targets antes de juntarlas...")
            # print("FEATURES (sample):")
            # print(featuresFichero3Elegidas.head())
            print("featuresFichero3Elegidas: " + str(featuresFichero3Elegidas.shape[0]) + " x " + str(
                featuresFichero3Elegidas.shape[1]))
            # print("TARGETS (sample):")
            # print(targetsFichero.head())

            featuresytargets = pd.concat(
                [featuresFichero3Elegidas.reset_index(drop=True), targetsFichero.reset_index(drop=True)],
                axis=1)  # Column bind
            featuresytargets.set_index(featuresFichero3Elegidas.index, inplace=True)
            # print("FEATURES+TARGETS juntas (sample):")
            # print(featuresytargets.head())
            print("Justo antes de guardar, featuresytargets: " + str(featuresytargets.shape[0]) + " x " + str(
                featuresytargets.shape[1]))
            featuresytargets.to_csv(pathCsvReducido, index=True, sep='|')


print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " ------------ FIN de capa 5 ----------------")

# GANADOR DEL SUBGRUPO (acumuladores)
ganador_nombreModelo = "NINGUNO"
ganador_metrica = 0
ganador_metrica_avg = 0
ganador_grid_mejores_parametros = []
pathListaColumnasCorreladasDrop = (dir_subgrupo + "columnas_correladas_drop" + ".txt").replace("futuro",
                                                                                               "pasado")  # lo guardamos siempre en el pasado

if (modoTiempo == "pasado" and pathCsvReducido.endswith('.csv') and os.path.isfile(pathCsvReducido) and os.stat(
        pathCsvReducido).st_size > 0):

    print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " Capa 6 - Modo PASADO")

    inputFeaturesyTarget = pd.read_csv(pathCsvReducido, index_col=0, sep='|')  # La columna 0 contiene el indice
    print("inputFeaturesyTarget: " + str(inputFeaturesyTarget.shape[0]) + " x " + str(inputFeaturesyTarget.shape[1]))

    print("BALANCEAR los casos positivos y negativos, haciendo downsampling de la clase mayoritaria...")
    print("URL: https://elitedatascience.com/imbalanced-classes")
    ift_mayoritaria = inputFeaturesyTarget[
        inputFeaturesyTarget.TARGET == False]  # En este caso los mayoritarios son los False
    ift_minoritaria = inputFeaturesyTarget[inputFeaturesyTarget.TARGET == True]
    print("ift_mayoritaria:" + str(ift_mayoritaria.shape[0]) + " x " + str(ift_mayoritaria.shape[1]))
    print("ift_minoritaria:" + str(ift_minoritaria.shape[0]) + " x " + str(ift_minoritaria.shape[1]))

    tasaDesbalanceoAntes = ift_mayoritaria.shape[0] / ift_minoritaria.shape[0]
    print("Tasa de desbalanceo entre clases (antes de balancear INICIO) = " + str(ift_mayoritaria.shape[0]) + "/" + str(
        ift_minoritaria.shape[0]) + " = " + str(tasaDesbalanceoAntes))
    num_muestras_minoria = ift_minoritaria.shape[0]

    casosInsuficientes = (num_muestras_minoria < umbralCasosSuficientesClasePositiva)
    if (casosInsuficientes):
        print("Numero de casos en clase minoritaria es INSUFICIENTE: " + str(num_muestras_minoria) + " (umbral=" + str(
            umbralCasosSuficientesClasePositiva) + "). Así que abandonamos este dataset y seguimos")

    else:
        ift_juntas = pd.concat([ift_mayoritaria.reset_index(drop=True), ift_minoritaria.reset_index(drop=True)],
                               axis=0)  # Row bind
        indices_juntos = ift_mayoritaria.index.append(ift_minoritaria.index)  # Row bind
        ift_juntas.set_index(indices_juntos, inplace=True)
        print("Las clases juntas son:")
        print("ift_juntas:" + str(ift_juntas.shape[0]) + " x " + str(ift_juntas.shape[1]))

        ############ PANDAS PROFILING ###########
        if modoDebug:
            print("REDUCIDO - Profiling...")
            if len(ift_juntas) > 2000:
                prof = ProfileReport(ift_juntas.drop(columns=['TARGET']).sample(n=2000))
            else:
                prof = ProfileReport(ift_juntas.drop(columns=['TARGET']))

            prof.to_file(output_file=dir_subgrupo + "REDUCIDO_profiling.html")

        ###################### Matriz de correlaciones y quitar features correladas ###################
        print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " Matriz de correlaciones (PASADO):")
        matrizCorr = ift_juntas.corr().abs()
        # print(matrizCorr.to_string())
        upper = matrizCorr.where(np.triu(np.ones(matrizCorr.shape), k=1).astype(np.bool))
        # print(upper.to_string())
        print("Eliminamos las features muy correladas (umbral =" + str(umbralFeaturesCorrelacionadas) + "):")
        to_drop = [column for column in upper.columns if any(upper[column] > umbralFeaturesCorrelacionadas)]
        print(to_drop)
        print("Guardamos esa lista de features muy correladas en: " + pathListaColumnasCorreladasDrop)
        pickle.dump(to_drop, open(pathListaColumnasCorreladasDrop, 'wb'))
        ift_juntas.drop(to_drop, axis=1, inplace=True)
        # print(ift_juntas)
        print("Matriz de correlaciones corregida, habiendo quitado las muy correlacionadas (PASADO):")
        matrizCorr = ift_juntas.corr()
        # print(matrizCorr)
        print("matrizCorr:" + str(matrizCorr.shape[0]) + " x " + str(matrizCorr.shape[1]))
        ##################################################################
        columnasSeleccionadas = ift_juntas.columns
        print("Guardando las columnas seleccionadas en: ", pathFeaturesSeleccionadas)
        print(columnasSeleccionadas)
        columnasSeleccionadasStr = '|'.join(columnasSeleccionadas)
        featuresSeleccionadasFile = open(pathFeaturesSeleccionadas, "w")
        featuresSeleccionadasFile.write(columnasSeleccionadasStr)
        featuresSeleccionadasFile.close()
        ##################################################################

        print((datetime.datetime.now()).strftime(
            "%Y%m%d_%H%M%S") + " DIVIDIR EL DATASET DE ENTRADA EN 3 PARTES: TRAIN (" + str(
            fraccion_train) + "), TEST (" + str(fraccion_test) + "), VALIDACION (" + str(fraccion_valid) + ")")
        ds_train, ds_test, ds_validacion = np.split(ift_juntas.sample(frac=1),
                                                    [int(fraccion_train * len(ift_juntas)),
                                                     int((fraccion_train + fraccion_test) * len(ift_juntas))])
        print("TRAIN = " + str(ds_train.shape[0]) + " x " + str(ds_train.shape[1]) + "  " + "TEST --> " + str(
            ds_test.shape[0]) + " x " + str(ds_test.shape[1]) + "  " + "VALIDACION --> " + str(
            ds_validacion.shape[0]) + " x " + str(ds_validacion.shape[1]))

        # Con las siguientes 3 líneas dejo sólo 1 true y lo demás false en el dataframe de test. Asi en la matriz de confusion validamos si el sistema no añade falsos positivos
        # aux = ds_test[ds_test.TARGET == True]
        # ds_test = ds_test[ds_test.TARGET == False]
        # ds_test=ds_test.append(aux.iloc[0])

        print("Separamos FEATURES y TARGETS, de los 3 dataframes...")
        ds_train_f = ds_train.drop('TARGET', axis=1).to_numpy()
        ds_train_t = ds_train[['TARGET']].to_numpy().ravel()
        ds_test_f = ds_test.drop('TARGET', axis=1).to_numpy()
        ds_test_t = ds_test[['TARGET']].to_numpy().ravel()
        ds_validac_f = ds_validacion.drop('TARGET', axis=1).to_numpy()
        ds_validac_t = ds_validacion[['TARGET']].to_numpy().ravel()

        feature_names = ds_train.columns.drop('TARGET')

        ########################### SMOTE-ENN (Oversampling con SMOTE y undersampling con Edited Nearest Neighbours) ##################

        df_mayoritaria = ds_train_t[ds_train_t == False]  # En este caso los mayoritarios son los False
        df_minoritaria = ds_train_t[ds_train_t == True]
        print("df_mayoritaria:" + str(len(df_mayoritaria)))
        print("df_minoritaria:" + str(len(df_minoritaria)))
        tasaDesbalanceoAntes = len(df_mayoritaria) / len(df_minoritaria)
        print("Tasa de desbalanceo entre clases (antes de balancear con SMOTE) = " + str(
            len(df_mayoritaria)) + " / " + str(len(df_minoritaria)) + " = " + str(tasaDesbalanceoAntes))

        balancearConSmoteSoloTrain = (
                    tasaDesbalanceoAntes > umbralNecesarioCompensarDesbalanceo)  # Condicion para decidir si hacer SMOTE
        if balancearConSmoteSoloTrain == True:
            print((datetime.datetime.now()).strftime(
                "%Y%m%d_%H%M%S") + " ---------------- RESAMPLING con SMOTE (porque supera umbral = " + str(
                umbralNecesarioCompensarDesbalanceo) + ") --------")
            print(
                "Resampling con SMOTE del vector de TRAINING (pero no a TEST ni a VALIDATION) según: " + "https://machinelearningmastery.com/combine-oversampling-and-undersampling-for-imbalanced-classification/")
            resample = SMOTEENN(sampling_strategy='minority',  # A que clase se hara el undersampling
                                random_state=0, smote=None, enn=None, n_jobs=-1)
            print("SMOTE antes (mayoritaria + minoritaria): %d" % ds_train_f.shape[0])
            ds_train_f, ds_train_t = resample.fit_resample(ds_train_f, ds_train_t)
            print("SMOTE después (mayoritaria + minoritaria): %d" % ds_train_f.shape[0])

            # print("Antes (mayoritaria + minoritaria): %d" % ds_train_f.shape[0])
            # # https://machinelearningmastery.com/undersampling-algorithms-for-imbalanced-classification/
            # undersample = NeighbourhoodCleaningRule(n_neighbors=3, threshold_cleaning=0.5)
            # ds_train_f, ds_train_t = undersample.fit_resample(ds_train_f, ds_train_t)
            # print("después (mayoritaria + minoritaria): %d" % ds_train_f.shape[0])
        ##################################################################

        ift_mayoritaria_entrada_modelos = ds_train_t[ds_train_t == False]  # En este caso los mayoritarios son los False
        ift_minoritaria_entrada_modelos = ds_train_t[ds_train_t == True]
        print("ift_mayoritaria_entrada_modelos:" + str(len(ift_mayoritaria_entrada_modelos)))
        print("ift_minoritaria_entrada_modelos:" + str(len(ift_minoritaria_entrada_modelos)))
        tasaDesbalanceoDespues = len(ift_mayoritaria_entrada_modelos) / len(ift_minoritaria_entrada_modelos)
        print("Tasa de desbalanceo entre clases (entrada a los modelos predictivos) = " + str(tasaDesbalanceoDespues))
        ##################################################################

        print("---------------- MODELOS con varias configuraciones (hiperparametros) --------")

        print("MODELOS: " + "https://scikit-learn.org/stable/supervised_learning.html")
        print(
            "EVALUACION de los modelos con: " + "https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics")
        print(
            "EVALUACION con curva precision-recall: " + "https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html")

        seed = 112  # Random seed

        #################### OPTIMIZACION DE PARAMETROS DE XGBOOST ###############################################################
        # Descomentar para obtener los parámetros con optimización Bayesiana
        # IMPORTANTE: se debe instalar el paquete de bayes en Conda: conda install -c conda-forge bayesian-optimization
        # Se imprimirán en el log, pero debo luego meterlos manualmente en el modelo
        # IMPORTANTE: DEBEN RELLENARSE 2 VALORES POR CADA ATRIBUTO DE PBOUND
        # https://ayguno.github.io/curious/portfolio/bayesian_optimization.html

        print("Inicio del optimizador de parametros de XGBOOST...")


        def xgboost_hyper_param(max_depth, learning_rate, n_estimators, reg_alpha, min_child_weight, colsample_bytree,
                                gamma):
            """Crea un modelo XGBOOST con los parametros indicados en la entrada. Aplica el numero de iteraciones de cross-validation indicado
                """
            clf = XGBClassifier(max_depth=int(max_depth), learning_rate=learning_rate, n_estimators=int(n_estimators),
                                reg_alpha=reg_alpha, min_child_weight=int(min_child_weight),
                                colsample_bytree=colsample_bytree, gamma=gamma,
                                nthread=-1, objective='binary:logistic', seed=seed)
            # "precision" tiene peores resultados que f1_weighted
            return np.mean(cross_val_score(clf, ds_train_f, ds_train_t, cv=cv_todos, scoring='f1_weighted'))


        # alpha is a parameter for the gaussian process
        # Note that this is itself a hyperparameter that can be optimized.
        gp_params = {"alpha": 1e-10}

        # Añadir carpeta dinámicamente: https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
        sys.path.append('/bayes_opt')
        from bayes_opt import BayesianOptimization

        pbounds = {
            'max_depth': (5, 20),
            'learning_rate': (0, 1),
            'n_estimators': (10, 100),
            'reg_alpha': (0, 1),
            'min_child_weight': (1, 20),
            'colsample_bytree': (0.1, 1),
            'gamma': (0, 10)
        }

        optimizer = BayesianOptimization(f=xgboost_hyper_param, pbounds=pbounds, random_state=1,
                                         verbose=10)
        optimizer.maximize(init_points=3, n_iter=10, acq='ucb', kappa=3, **gp_params)
        valoresOptimizados = optimizer.max
        print(valoresOptimizados)
        print("Fin del optimizador")
        ###################################################################################

        print("Inicio de XGBOOST")
        nombreModelo = "xgboost"
        pathModelo = dir_subgrupo + nombreModelo + ".modelo"
        # Parametros: https://xgboost.readthedocs.io/en/latest/parameter.html
        # Instalación en Conda: conda install -c anaconda py-xgboost
        # Instalación en Python básico: pip install xgboost

        # MODELO LUIS AUTOOPTIMIZADO PARA CADA SUBGRUPO
        max_depth = int(valoresOptimizados.get("params").get("max_depth"))
        learning_rate = valoresOptimizados.get("params").get("learning_rate")
        n_estimators = int(valoresOptimizados.get("params").get("n_estimators"))
        reg_alpha = valoresOptimizados.get("params").get("reg_alpha")
        min_child_weight = int(valoresOptimizados.get("params").get("min_child_weight"))
        colsample_bytree = valoresOptimizados.get("params").get("colsample_bytree")
        gamma = valoresOptimizados.get("params").get("gamma")
        nthread = -1
        objective = 'binary:logistic'
        seed = seed

        modeloXGB = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators,
                               reg_alpha=reg_alpha, min_child_weight=min_child_weight,
                               colsample_bytree=colsample_bytree, gamma=gamma,
                               nthread=nthread, objective=objective, seed=seed)

        print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " Ejecutando " + nombreModelo + " ...")
        out_grid_best_params = []

        param_parada_iteraciones = 10  # early_stopping_rounds: es el numero de iteraciones en las que ya no mejora el error diferencial train-test, evitando iterar tanto en XGBoost y reducir el overfitting
        eval_set = [(ds_train_f, ds_train_t), (ds_test_f, ds_test_t)]

        # -------- PINTAR EL ERROR DE OVERFITTING ---------------------------
        # -------------------URL: https://machinelearningmastery.com/avoid-overfitting-by-early-stopping-with-xgboost-in-python/
        # --- URL: https://xgboost.readthedocs.io/en/latest/parameter.html

        METODO_EVALUACION = "map"  # map: Mean average Precision. aucpr: Area under the PR curve (peores resultados en precisión)
        # Explicacion: https://xgboost.readthedocs.io/en/latest/parameter.html

        # Con PARAMETROS PARA VER EL OVERFITTING
        modelo = modeloXGB.fit(ds_train_f, ds_train_t, eval_metric=[METODO_EVALUACION],
                            early_stopping_rounds=param_parada_iteraciones, eval_set=eval_set,
                            verbose=False)  # ENTRENAMIENTO (TRAIN)

        # --------------- Pintar dibujo---------------------------------------------------------------
        y_pred = modelo.predict(ds_test_f)
        y_pred = y_pred.astype(float)
        predictions = [round(value) for value in y_pred]
        precision_para_medir_overfitting = precision_score(ds_test_t, predictions)
        print("Accuracy (PRECISION) para medir el overfitting: %.2f%%" % (precision_para_medir_overfitting * 100.0))
        results = modelo.evals_result()

        epochs = len(results['validation_0'][METODO_EVALUACION])
        x_axis = range(0, epochs)
        fig, ax = pyplot.subplots()
        ax.plot(x_axis, results['validation_0'][METODO_EVALUACION], label='Train')
        ax.plot(x_axis, results['validation_1'][METODO_EVALUACION], label='Test')
        ax.legend()
        pyplot.xlabel("Numero de epochs")
        pyplot.ylabel(METODO_EVALUACION)
        pyplot.title("Modelo: " + nombreModelo + " - Metodo de evaluacion: " + METODO_EVALUACION)
        path_img_metricas_modelo_ovft = dir_subgrupo_img + nombreModelo + "_" + METODO_EVALUACION + ".png"
        print(
            "Pintando IMG de metricas del modelo overfitting (train vs test). Path: " + path_img_metricas_modelo_ovft)
        plt.savefig(path_img_metricas_modelo_ovft, bbox_inches='tight')
        plt.clf();
        plt.cla();
        plt.close();  # Limpiando dibujo
        # ------------------------------------------------------------------------------

        pickle.dump(modelo, open(pathModelo, 'wb'))

        modelo_xgboost = modelo

        modelo_loaded = pickle.load(open(pathModelo, 'rb'))
        ds_test_t_pred = modelo_loaded.predict(
            ds_test_f)  # PREDICCION de los targets de TEST (los compararemos con los que tenemos)


        test_t_predicho = modelo_xgboost.predict(ds_test_f);
        validac_t_predicho = modelo_xgboost.predict(ds_validac_f)
        precision_test = precision_score(ds_test_t, test_t_predicho);
        precision_avg_test = average_precision_score(ds_test_t, test_t_predicho)
        precision_validation = precision_score(ds_validac_t, validac_t_predicho);
        precision_avg_validation = average_precision_score(ds_validac_t, validac_t_predicho)
        precision_media = (precision_test + precision_validation) / 2
        precision_avg_media = (precision_avg_test + precision_avg_validation) / 2
        print(id_subgrupo + " " + nombreModelo + " -> Precision = " + '{0:0.2f}'.format(
            precision_media) + " (average precision = " + '{0:0.2f}'.format(precision_avg_media) + ")")
        if precision_media > ganador_metrica:
            ganador_metrica = precision_media
            ganador_metrica_avg = precision_avg_media
            ganador_nombreModelo = nombreModelo
            ganador_grid_mejores_parametros = []  # Este modelo no tiene esta variable
        print("Fin de XGBOOST")

        ######################################################################################################################
        ######################################################################################################################

        print("********* GANADOR de subgrupo *************")
        print("PASADO -> " + id_subgrupo + " (num features = " + str(
            ds_train_f.shape[1]) + ")" + " -> Modelo ganador = " + ganador_nombreModelo + " --> METRICA = " + str(
            round(ganador_metrica, 4)) + " (avg_precision = " + str(round(ganador_metrica_avg, 4)) + ") ")
        print("Hiperparametros:")
        print(ganador_grid_mejores_parametros)
        pathModeloGanadorDeSubgrupoOrigen = dir_subgrupo + ganador_nombreModelo + ".modelo"
        pathModeloGanadorDeSubgrupoDestino = pathModeloGanadorDeSubgrupoOrigen + "_ganador"
        copyfile(pathModeloGanadorDeSubgrupoOrigen, pathModeloGanadorDeSubgrupoDestino)
        print("Modelo ganador guardado en: " + pathModeloGanadorDeSubgrupoDestino)

        ######################################################################################################################
        ######################################################################################################################


elif (modoTiempo == "futuro" and pathCsvReducido.endswith('.csv') and os.path.isfile(pathCsvReducido) and os.stat(
        pathCsvReducido).st_size > 0):

    print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " Capa 6 - Modo futuro")
    inputFeaturesyTarget = pd.read_csv(pathCsvReducido, index_col=0, sep='|')
    print("inputFeaturesyTarget: " + str(inputFeaturesyTarget.shape[0]) + " x " + str(inputFeaturesyTarget.shape[1]))

    print("Si las hay, eliminamos las features muy correladas (umbral =" + str(
        umbralFeaturesCorrelacionadas) + ") aprendido en el PASADO..")
    if os.path.exists(pathListaColumnasCorreladasDrop):
        columnasCorreladas = pickle.load(open(pathListaColumnasCorreladasDrop, 'rb'))
        inputFeaturesyTarget.drop(columnasCorreladas, axis=1, inplace=True)
        print(columnasCorreladas)

    print("Matriz de correlaciones corregida (FUTURO):")
    matrizCorr = inputFeaturesyTarget.corr()
    print(matrizCorr)

    print(
        "La columna TARGET que haya en el CSV de entrada no la queremos (es un NULL o False, por defecto), porque la vamos a PREDECIR...")
    inputFeatures = inputFeaturesyTarget.drop('TARGET', axis=1)
    print(inputFeatures.head())
    print("inputFeatures: " + str(inputFeatures.shape[0]) + " x " + str(inputFeatures.shape[1]))

    print(
        "MISSING VALUES (FILAS) - Borramos las FILAS que tengan 1 o mas valores NaN porque son huecos que no deberían estar...")
    inputFeatures_sinnulos = inputFeatures.dropna(axis=0, how='any')  # Borrar FILA si ALGUNO sus valores tienen NaN

    dir_modelo_predictor_ganador = dir_subgrupo.replace("futuro",
                                                        "pasado")  # Siempre cojo el modelo entrenado en el pasado
    path_modelo_predictor_ganador = ""
    for file in os.listdir(dir_modelo_predictor_ganador):
        if file.endswith("ganador"):
            path_modelo_predictor_ganador = os.path.join(dir_modelo_predictor_ganador, file)

    print("Cargar modelo PREDICTOR ganador (de la carpeta del pasado, SI EXISTE): " + path_modelo_predictor_ganador)
    if os.path.isfile(path_modelo_predictor_ganador):

        modelo_predictor_ganador = pickle.load(open(path_modelo_predictor_ganador, 'rb'))

        print("Predecir:")
        cols_when_model_builds = modelo_predictor_ganador.get_booster().feature_names
        # El modelo XGB guarda como nombres de los features: fx, donde x=0, 1, 2... Si es así, se cambiará el
        # nombre de los features de entrada al predictor
        i = 0
        inputFeatures_sinnulosNuevo = pd.DataFrame()
        if 'f0' in cols_when_model_builds:
            for nombreFeature in inputFeatures_sinnulos.columns:
                nuevoNombre = 'f' + str(i)
                print("Antes de predecir se sustituye " + nombreFeature + " por " + nuevoNombre)
                inputFeatures_sinnulosNuevo[nuevoNombre] = inputFeatures_sinnulos[nombreFeature]
                i += 1
            inputFeatures_sinnulos = inputFeatures_sinnulosNuevo

        targets_predichos = modelo_predictor_ganador.predict(inputFeatures_sinnulos)
        num_targets_predichos = len(targets_predichos)
        print("Numero de targets_predichos: " + str(num_targets_predichos) + " con numero de TRUEs = " + str(
            np.sum(targets_predichos, where=["True"])))
        # print("El array de targets contiene:")
        # print(targets_predichos)

        # probabilities
        probs = pd.DataFrame(data=modelo_predictor_ganador.predict_proba(inputFeatures_sinnulos),
                             index=inputFeatures_sinnulos.index)

        # UMBRAL MENOS PROBABLES CON TARGET=1. Cuando el target es 1, se guarda su probabilidad
        # print("El DF llamado probs contiene las probabilidades de predecir un 0 o un 1:")
        # print(probs)

        probabilidadesEnTargetUnoPeq = probs.iloc[:, 1]  # Cogemos solo la segunda columna: prob de que sea target=1
        probabilidadesEnTargetUnoPeq2 = probabilidadesEnTargetUnoPeq.apply(
            lambda x: x if (x >= umbralProbTargetTrue) else np.nan)  # Cogemos solo las filas cuya prob_1 > umbral
        probabilidadesEnTargetUnoPeq3 = probabilidadesEnTargetUnoPeq2[
            np.isnan(probabilidadesEnTargetUnoPeq2[:]) == False]  # Cogemos todos los no nulos (NAN)
        # print("El DF llamado probabilidadesEnTargetUnoPeq3 contiene las probabilidades de los UNO con prob mayor que umbral ("+str(umbralProbTargetTrue)+"):")
        # print(probabilidadesEnTargetUnoPeq3)

        probabilidadesEnTargetUnoPeq4 = probabilidadesEnTargetUnoPeq3.sort_values(ascending=False)  # descendente
        print(
            "El DF llamado probabilidadesEnTargetUnoPeq4 contiene los indices y probabilidades, tras aplicar umbral INFERIOR: " + str(
                umbralProbTargetTrue) + ". Son:")
        print(probabilidadesEnTargetUnoPeq4)

        numfilasSeleccionadas = int(granProbTargetUno * probabilidadesEnTargetUnoPeq4.shape[
            0] / 100)  # Como están ordenadas en descendente, cojo estas NUM primeras filas
        print("numfilasSeleccionadas: " + str(numfilasSeleccionadas))
        targets_predichosCorregidos_probs = probabilidadesEnTargetUnoPeq4[0:numfilasSeleccionadas]
        targets_predichosCorregidos = targets_predichosCorregidos_probs.apply(lambda x: 1)
        print(
            "El DF llamado targets_predichosCorregidos contiene los indices y probabilidades, tras aplicar umbral SUPERIOR: top " + str(
                granProbTargetUno) + " % de muestras. Son:")
        print(targets_predichosCorregidos)

        print("Guardando targets PREDICHOS en: " + pathCsvPredichos)
        df_predichos = targets_predichosCorregidos.to_frame()
        df_predichos.columns = ['TARGET_PREDICHO']
        df_predichos.to_csv(pathCsvPredichos, index=False, sep='|')  # Capa 6 - Salida (para el validador, sin indice)

        df_predichos_probs = targets_predichosCorregidos_probs.to_frame()
        df_predichos_probs.columns = ['TARGET_PREDICHO_PROB']
        df_predichos_probs.to_csv(pathCsvPredichos + "_humano", index=True, sep='|')  # Capa 6 - Salida (para el humano)

        ############### RECONSTRUCCION DEL CSV FINAL IMPORTANTE, viendo los ficheros de indices #################
        print(
            "Partiendo de COMPLETO.csv llevamos la cuenta de los indices pasando por REDUCIDO.csv y por TARGETS_PREDICHOS.csv para generar el CSV final...")
        df_completo = pd.read_csv(pathCsvCompleto, sep='|')  # Capa 5 - Entrada

        print("df_completo: " + str(df_completo.shape[0]) + " x " + str(df_completo.shape[1]))
        print("df_predichos: " + str(df_predichos.shape[0]) + " x " + str(df_predichos.shape[1]))
        print("df_predichos_probs: " + str(df_predichos_probs.shape[0]) + " x " + str(df_predichos_probs.shape[1]))

        print("Juntar COMPLETO con TARGETS PREDICHOS... ")
        df_juntos_1 = pd.concat([df_completo, df_predichos], axis=1)
        df_juntos_2 = pd.concat([df_juntos_1, df_predichos_probs], axis=1)

        df_juntos_2['TARGET_PREDICHO'] = (df_juntos_2['TARGET_PREDICHO'] * 1).astype(
            'Int64')  # Convertir de boolean a int64, manteniendo los nulos

        print("Guardando: " + pathCsvFinalFuturo)
        df_juntos_2.to_csv(pathCsvFinalFuturo, index=False, sep='|')


    else:
        print(
            "No existe el modelo predictor del pasado que necesitamos (" + path_modelo_predictor_ganador + "). Por tanto, no predecimos.")


else:
    print("Los parametros de entrada son incorrectos o el CSV no existe o esta vacio!!")

############################################################
print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " ------------ FIN de capa 6----------------")
