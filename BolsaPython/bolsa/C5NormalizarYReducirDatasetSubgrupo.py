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


print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " **** CAPA 5  --> Selección de variables/ Reducción de dimensiones (para cada subgrupo) ****")
print("URL PCA: https://scikit-learn.org/stable/modules/unsupervised_reduction.html")
print("URL Feature Selection: https://scikit-learn.org/stable/modules/feature_selection.html")

##################################################################################################
print("PARAMETROS: ")
dir_subgrupo = sys.argv[1]
modoTiempo = sys.argv[2]
maxFeatReducidas = sys.argv[3]
maxFilasEntrada = sys.argv[4]
print("dir_subgrupo = %s" % dir_subgrupo)
print("modoTiempo = %s" % modoTiempo)
print("maxFeatReducidas = %s" % maxFeatReducidas)
print("maxFilasEntrada = %s" % maxFilasEntrada)

varianza = 0.92  # Variacion acumulada de las features PCA sobre el target
compatibleParaMuchasEmpresas = False  # Si hay muchas empresas, debo hacer ya el undersampling (en vez de capa 6)
global modoDebug; modoDebug = False  # VARIABLE GLOBAL: En modo debug se pintan los dibujos. En otro caso, se evita calculo innecesario
global cv_todos; cv_todos = 15  # CROSS_VALIDATION: número de iteraciones. Sirve para evitar el overfitting
global rfecv_step; rfecv_step = 3  # Numero de features que va reduciendo en cada iteracion de RFE hasta encontrar el numero deseado
global dibujoBins; dibujoBins = 20  # VARIABLE GLOBAL: al pintar los histogramas, define el número de barras posibles en las que se divide el eje X.
numTramos = 7  # Numero de tramos usado para tramificar las features dinámicas
pathCsvCompleto = dir_subgrupo + "COMPLETO.csv"
dir_subgrupo_img = dir_subgrupo + "img/"
pathCsvIntermedio = dir_subgrupo + "intermedio.csv"
pathCsvReducido = dir_subgrupo + "REDUCIDO.csv"
pathCsvFeaturesElegidas = dir_subgrupo + "FEATURES_ELEGIDAS_RFECV.csv"
pathModeloOutliers = (dir_subgrupo + "DETECTOR_OUTLIERS.tool").replace("futuro", "pasado")  # Siempre lo cojo del pasado
path_modelo_tramificador = (dir_subgrupo + "tramif/" + "TRAMIFICADOR").replace("futuro", "pasado")  # Siempre lo cojo del pasado
path_modelo_normalizador = (dir_subgrupo + "NORMALIZADOR.tool").replace("futuro", "pasado")  # Siempre lo cojo del pasado
path_indices_out_capa5 = (dir_subgrupo + "indices_out_capa5.indices")
path_modelo_reductor_features = (dir_subgrupo + "REDUCTOR.tool").replace("futuro", "pasado")  # Siempre lo cojo del pasado
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
print("balancear = " + str(balancear))


######################## FUNCIONES #######################################################

with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore", message="Bins whose width are too small.*")  # Ignorar los warnings del tramificador (KBinsDiscretizer)


def leerFeaturesyTarget(path_csv_completo, path_dir_img, compatibleParaMuchasEmpresas, pathModeloOutliers, modoTiempo, maxFilasEntrada):
  print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " ----- leerFeaturesyTarget ------")
  print("PARAMS --> " + path_csv_completo + "|" + path_dir_img + "|" + str(compatibleParaMuchasEmpresas) + "|" + pathModeloOutliers + "|" + modoTiempo + "|" + str(modoDebug)
        +"|" + str(maxFilasEntrada))

  print("Cargar datos (CSV)...")
  entradaFeaturesYTarget = pd.read_csv(filepath_or_buffer=path_csv_completo, sep='|')
  print("entradaFeaturesYTarget (LEIDO): " + str(entradaFeaturesYTarget.shape[0]) + " x " + str(entradaFeaturesYTarget.shape[1]))
  entradaFeaturesYTarget.sort_index(inplace=True)  # Reordenando segun el indice (para facilitar el testIntegracion)
  entradaFeaturesYTarget.to_csv(pathCsvIntermedio + ".entrada", index=True, sep='|')  # NO BORRAR: UTIL para testIntegracion

  if int(maxFilasEntrada) < entradaFeaturesYTarget.shape[0]:
      print("entradaFeaturesYTarget (APLICANDO MAXIMO): " + str(entradaFeaturesYTarget.shape[0]) + " x " + str(
          entradaFeaturesYTarget.shape[1]))
      entradaFeaturesYTarget = entradaFeaturesYTarget.sample(int(maxFilasEntrada), replace=False)

  entradaFeaturesYTarget.sort_index(inplace=True)  # Reordenando segun el indice (para facilitar el testIntegracion)
  entradaFeaturesYTarget.to_csv(pathCsvIntermedio + ".entrada_tras_maximo.csv", index=True, sep='|')  # NO BORRAR: UTIL para testIntegracion
  entradaFeaturesYTarget.to_csv(pathCsvIntermedio + ".entrada_tras_maximo_INDICES.csv", columns=[])  # NO BORRAR: UTIL para testIntegracion

  num_nulos_por_fila_1 = np.logical_not(entradaFeaturesYTarget.isnull()).sum()
  indiceFilasFuturasTransformadas1 = entradaFeaturesYTarget.index.values  # DEFAULT
  indiceFilasFuturasTransformadas2 = entradaFeaturesYTarget.index.values  # DEFAULT

  ################# Borrado de columnas nulas enteras ##########
  print("MISSING VALUES (COLUMNAS) - Borramos las columnas (features) que sean siempre NaN...")
  entradaFeaturesYTarget2 = entradaFeaturesYTarget.dropna(axis=1, how='all') #Borrar COLUMNA si TODOS sus valores tienen NaN
  print("entradaFeaturesYTarget2 (columnas nulas borradas): " + str(entradaFeaturesYTarget2.shape[0]) + " x " + str(entradaFeaturesYTarget2.shape[1]))
  # entradaFeaturesYTarget2.to_csv(path_csv_completo + "_TEMP01", index=True, sep='|')  # UTIL para testIntegracion


  ################# Borrado de filas que tengan algun hueco (tratamiento espacial para el futuro con sus columnas: TARGET y otras) #####
  print("MISSING VALUES (FILAS)...")
  ########## IMPUTACION ##########
  # print("IMPUTACION de valores donde había NAN ")
  # nombres_columnas_numericas = entradaFeaturesYTarget2.select_dtypes(include=np.number).columns.tolist()
  # nombres_columnas_no_numericas = entradaFeaturesYTarget2.select_dtypes(exclude=np.number).columns.tolist()
  # entradaFeaturesYTarget2_num = entradaFeaturesYTarget2.drop(nombres_columnas_no_numericas, axis=1)
  # entradaFeaturesYTarget2_nonum = entradaFeaturesYTarget2.drop(nombres_columnas_numericas, axis=1)
  #
  # my_imputer = SimpleImputer(strategy='median')
  # entradaFeaturesYTarget2_imputado = pd.DataFrame(my_imputer.fit_transform(entradaFeaturesYTarget2_num))
  # entradaFeaturesYTarget2_imputado.columns = entradaFeaturesYTarget2_num.columns  # La imputacion borró los nombres de columnas. Los reponemos.
  # entradaFeaturesYTarget2 = pd.concat([entradaFeaturesYTarget2_nonum, entradaFeaturesYTarget2_imputado], axis=1)
  # print("entradaFeaturesYTarget2 (filas con algun nulo IMPUTADAS): " + str(entradaFeaturesYTarget2.shape[0]) + " x " + str(entradaFeaturesYTarget2.shape[1]))
  ################################

  if modoTiempo == "futuro":
      print("Nos quedamos solo con las velas con antiguedad=0 (futuras)...")
      entradaFeaturesYTarget2 = entradaFeaturesYTarget2[entradaFeaturesYTarget2.antiguedad == 0]
      print("entradaFeaturesYTarget2:" + str(entradaFeaturesYTarget2.shape[0]) + " x " + str(entradaFeaturesYTarget2.shape[1]))
      #print(entradaFeaturesYTarget2.head())

  print("Porcentaje de MISSING VALUES en cada columna del dataframe de entrada (mostramos las que superen 20%):")
  missing = pd.DataFrame(entradaFeaturesYTarget2.isnull().sum()).rename(columns={0: 'total'})
  missing['percent'] = missing['total'] / len(entradaFeaturesYTarget2)  # Create a percentage missing
  missing_df = missing.sort_values('percent', ascending=False)
  missing_df = missing_df[missing_df['percent'] > 0.20]
  print(missing_df.to_string())  # .drop('TARGET')

  # print("Pasado o Futuro: Transformacion en la que borro filas. Por tanto, guardo el indice...")
  indiceFilasFuturasTransformadas1 = entradaFeaturesYTarget2.index.values

  print("Borrar columnas especiales (idenficadoras de fila): empresa | antiguedad | mercado | anio | mes | dia | hora | minuto...")
  entradaFeaturesYTarget2 = entradaFeaturesYTarget2\
      .drop('empresa', axis=1).drop('antiguedad', axis=1).drop('mercado', axis=1)\
      .drop('anio', axis=1).drop('mes', axis=1).drop('dia', axis=1).drop('hora', axis=1).drop('minuto', axis=1)\

  print(
      "Borrar columnas dinamicas que no aportan nada: volumen | high | low | close | open ...")
  entradaFeaturesYTarget2 = entradaFeaturesYTarget2 \
      .drop('volumen', axis=1).drop('high', axis=1).drop('low', axis=1).drop('close', axis=1).drop('open', axis=1)
  
  # entradaFeaturesYTarget2.to_csv(path_csv_completo + "_TEMP02", index=True, sep='|')  # UTIL para testIntegracion

  print("entradaFeaturesYTarget2: " + str(entradaFeaturesYTarget2.shape[0]) + " x " + str(entradaFeaturesYTarget2.shape[1]))
  # print(entradaFeaturesYTarget2.head())

  if modoTiempo == "pasado":
      print("MISSING VALUES (FILAS) - Borramos las FILAS que tengan 1 o mas valores NaN porque son huecos que no deberían estar...")
      entradaFeaturesYTarget3 = entradaFeaturesYTarget2.dropna(axis=0, how='any')  # Borrar FILA si ALGUNO sus valores tienen NaN

  elif modoTiempo == "futuro":
      print("MISSING VALUES (FILAS) - Para el FUTURO; el target es NULO siempre, pero borramos las filas que tengan ademas otros NULOS...")
      entradaFeaturesYTarget3 = entradaFeaturesYTarget2.drop('TARGET', axis=1).dropna(axis=0, how='any')  # Borrar FILA si ALGUNO sus valores tienen NaN


  print("Pasado o futuro: Transformacion en la que he borrado filas. Por tanto, guardo el indice...")
  indiceFilasFuturasTransformadas2 = entradaFeaturesYTarget3.index.values
  print("indiceFilasFuturasTransformadas2: " + str(indiceFilasFuturasTransformadas2.shape[0]))
  #print(indiceFilasFuturasTransformadas2)

  print("entradaFeaturesYTarget3 (filas con algun nulo borradas):" + str(entradaFeaturesYTarget3.shape[0]) + " x " + str(entradaFeaturesYTarget3.shape[1]))
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
  flagAnomaliasDf = pd.DataFrame({'marca_anomalia': detector_outliers.predict(df3aux)})  # vale -1 es un outlier; si es un 1, no lo es

  indice3=entradaFeaturesYTarget3.index  # lo guardo para pegarlo luego
  entradaFeaturesYTarget3.reset_index(drop=True, inplace=True)
  flagAnomaliasDf.reset_index(drop=True, inplace=True)
  entradaFeaturesYTarget4 = pd.concat([entradaFeaturesYTarget3, flagAnomaliasDf], axis=1)  # Column Bind, manteniendo el índice del DF izquierdo
  entradaFeaturesYTarget4.set_index(indice3, inplace=True)  # ponemos el indice que tenia el DF de la izquierda

  entradaFeaturesYTarget4 = entradaFeaturesYTarget4.loc[entradaFeaturesYTarget4['marca_anomalia'] == 1] #Cogemos solo las que no son anomalias
  entradaFeaturesYTarget4 = entradaFeaturesYTarget4.drop('marca_anomalia', axis=1)  #Quitamos la columna auxiliar

  print("entradaFeaturesYTarget4 (sin outliers):" + str(entradaFeaturesYTarget4.shape[0]) + " x " + str(entradaFeaturesYTarget4.shape[1]))
  # entradaFeaturesYTarget4.to_csv(path_csv_completo + "_TEMP04", index=True, sep='|')  # UTIL para testIntegracion

  # ---------------------------------------------
  # # Nota: he visto que no sale ninún caso en lo nuestro, así que lo comento, para no complicar el código
  # # Se eliminan todas las features que sean poco significativas (es decir, si contienen un único valor
  # # en más del xx% de los casos).
  # # No se tiene en cuenta la columna TARGET (que casi seguro que está presente)
  # # Se sigue el criterio de: https://www.kaggle.com/jesucristo/1-house-prices-solution-top-1
  # overfit = []
  # for i in entradaFeaturesYTarget4.columns:
  #     counts = entradaFeaturesYTarget4[i].value_counts()
  #     zeros = counts.iloc[0]
  #     if zeros / len(entradaFeaturesYTarget4) > 0.8:
  #         overfit.append(i)
  # overfit = list(overfit)
  # overfit.remove("TARGET")
  # print("entradaFeaturesYTarget4 antes de quitar filas no significativas: " + str(entradaFeaturesYTarget4.shape[0]))
  # entradaFeaturesYTarget4 = entradaFeaturesYTarget4.drop(overfit, axis=1)
  # print("entradaFeaturesYTarget4 después de quitar filas no significativas: " + str(entradaFeaturesYTarget4.shape[0]))

  # ---------------------------------------------
  # #KMEANS
  # # Clusterizamos los datos del PASADO con target=1, y nos quedamos sólo con el cluster con más elementos. Al resto
  # # de clusters, los ponemos target=0
  # if modoTiempo == "pasado":
  #     # Busco el mejor valor de K para k-means
  #     filasConTargetUno = entradaFeaturesYTarget4.loc[entradaFeaturesYTarget4['TARGET'] == 1]
  #     filasConTargetCeros = entradaFeaturesYTarget4.loc[entradaFeaturesYTarget4['TARGET'] == 0]
  #     print("filasConTargetUno: " + str(filasConTargetUno.shape[0]))
  #     print("filasConTargetCero: " + str(filasConTargetCeros.shape[0]))
  #     distortions = []
  #     K = range(1, 10)
  #     for k in K:
  #         kmeanModel = KMeans(n_clusters=k).fit(filasConTargetUno)
  #         kmeanModel.fit(filasConTargetUno)
  #         distortions.append(sum(np.min(cdist(filasConTargetUno, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) /
  #                            filasConTargetUno.shape[0])
  #
  #     # El k óptimo será el de menor distorsión respecto a su valor anterior
  #     distorsionAnterior = 1000000
  #     variacionAnterior = 0
  #     kOptimo = 2
  #     difAnterior = 0
  #     for dist in distortions:
  #         variacion = distorsionAnterior / (distorsionAnterior - dist)
  #         dif = variacionAnterior - variacion
  #         if kOptimo > 2:
  #             if (difAnterior - dif) < dif:
  #                 kOptimo += 1
  #             else:
  #                 break
  #         else:
  #             kOptimo += 1
  #         variacionAnterior = variacion
  #         difAnterior = dif
  #     # # Plot the elbow
  #     # plt.plot(K, distortions, 'bx-')
  #     # plt.xlabel('k')
  #     # plt.ylabel('Distortion')
  #     # plt.title('The Elbow Method showing the optimal k')
  #     # plt.show()
  #     # Ejecutamos K-MEANS con el k óptimo y lo aplicamos a los datos, para clasificar los tipos de targetUno
  #     # Nos quedamos con los targetUno del cluster más numeroso
  #     kmeans = KMeans(n_clusters=kOptimo).fit(filasConTargetUno)
  #     # # Centroides
  #     # centroids = kmeans.cluster_centers_
  #     # print(centroids)
  #     clusterIDs = kmeans.predict(filasConTargetUno)
  #     filasConTargetUno['clusterId'] = pd.Series(clusterIDs, index=filasConTargetUno.index)
  #     clusterMasNumeroso = most_frequent(list(clusterIDs))
  #     filasConTargetUnoClusterSelec = filasConTargetUno.loc[filasConTargetUno['clusterId'] == clusterMasNumeroso].drop('clusterId', axis=1)
  #     filasConTargetUnoClusterNoSelec = filasConTargetUno.loc[filasConTargetUno['clusterId'] != clusterMasNumeroso].drop(
  #         'clusterId', axis=1)
  #     print("El kOptimo de kmeans es: " + str(kOptimo))
  #     print("El cluster más numeroso tiene estos elementos: " + str(filasConTargetUnoClusterSelec.shape[0]))
  #     print("El resto de clusters juntos tiene estos elementos: " + str(filasConTargetUnoClusterNoSelec.shape[0]))
  #     # Los target Uno de los clusters NO seleccionados se quitan
  #     entradaFeaturesYTarget4=entradaFeaturesYTarget4.drop(filasConTargetUnoClusterNoSelec.index)
  # ---------------------------------------------

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

      if(balancear == True):
          print("BALANCEAR los casos positivos y negativos, haciendo downsampling de la clase mayoritaria...")
          print("URL: https://elitedatascience.com/imbalanced-classes")
          ift_minoritaria = entradaFeaturesYTarget4[entradaFeaturesYTarget4.TARGET == True]
          print("ift_minoritaria (original):" + str(ift_minoritaria.shape[0]) + " x " + str(ift_minoritaria.shape[1]))
          num_filas_azar = 5 * ift_minoritaria.shape[0]
          print("num_filas_azar:" + str(num_filas_azar))
          ift_mayoritaria = entradaFeaturesYTarget4.loc[np.random.choice(entradaFeaturesYTarget4.index, num_filas_azar)]
          ift_mayoritaria = ift_mayoritaria[ift_mayoritaria.TARGET == False]
          print("ift_mayoritaria (se han borrado filas, pero no muchas):" + str(ift_mayoritaria.shape[0]) + " x " + str(ift_mayoritaria.shape[1]))
          print("ift_minoritaria (con oversampling): " + str(ift_minoritaria.shape[0]) + " x " + str(ift_minoritaria.shape[1]))
          print("Tasa de desbalanceo entre clases = " + str(ift_mayoritaria.shape[0]) + "/" + str(ift_minoritaria.shape[0]) + " = " + str(ift_mayoritaria.shape[0] / ift_minoritaria.shape[0]))
          # Juntar ambas clases ya BALANCEADAS. Primero vacío el dataset
          entradaFeaturesYTarget5 = ift_mayoritaria.append(ift_minoritaria)
          print("Las clases ya están balanceadas:")
          print("ift_balanceadas:" + str(entradaFeaturesYTarget5.shape[0]) + " x " + str(entradaFeaturesYTarget5.shape[1]))

      else:
          print("NO balanceamos clases en capa 5 (pero seguramente sí en capa 6 solo sobre dataset de TRAIN)!!!")
          ift_minoritaria = entradaFeaturesYTarget4[entradaFeaturesYTarget4.TARGET == True]
          ift_mayoritaria = entradaFeaturesYTarget4[entradaFeaturesYTarget4.TARGET == False]
          print("Tasa de desbalanceo entre clases = " + str(ift_mayoritaria.shape[0]) + "/" + str(ift_minoritaria.shape[0]) + " = " + str(ift_mayoritaria.shape[0] / ift_minoritaria.shape[0]))
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
      path_dibujo = path_dir_img + column + ".png"
      print("Guardando distrib de col: " + column + " en fichero: " + path_dibujo)
      datos_columna = featuresFichero[column]
      sns.distplot(datos_columna, kde=False, color='red', bins=dibujoBins)
      plt.title(column, fontsize=10)
      plt.savefig(path_dibujo, bbox_inches='tight')
      plt.clf(); plt.cla(); plt.close()  # Limpiando dibujo

  return featuresFichero, targetsFichero


def tramificarFeatures(numTramos, featuresFichero, targetsFichero, path_modelo_tramificador, path_dir_img, modoTiempo):
    print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " -------- TRAMIFICAR FEATURES: análisis UNIVARIANTE (estudia CADA feature) ------")
    print("Info sobre discretizar: http://exponentis.es/discretizacion-de-datos-en-python-manteniendo-el-nombre-de-las-columnas")
    print("PASADO: para cada feature dinámica, tramificar, viendo en qué tramos caen los target=1 para que todos los tramos tengan ceros y unos, sin demasiado desbalanceo. Despues, guardar el discretizador para usarlo en el futuro")
    print("FUTURO: aplicar el discretizador de columnas que se haya usado en el pasado")

    print("Tramificador - Numero de tramos para cada feature: " + str(numTramos))
    cabecera = list(featuresFichero)  # Guardamos los nombres de las columnas.
    indice = -1  # Contador para iterar por columnas.

    if modoTiempo == "pasado":

        tramif_calidad_stddev_balanceotramo = pd.DataFrame(columns=['STD_DEV_del_pct_positivos_en_tramo', 'FEATURE'])

        while indice < (len(cabecera) - 1):
            indice = indice + 1

            print("Tramificando feature '" + cabecera[indice] + "' (indice: " + str(indice) + ")")
            featureAnalizada = featuresFichero[cabecera[indice]]  # Feature analizada
            featureAnalizada = featureAnalizada.to_frame()  # Conversion a dataframe
            featureAnalizada.set_index(featuresFichero.index)
            path_tramificador_feature = path_modelo_tramificador + "_FEAT_" + cabecera[indice]
            desv_std_feature = np.std(featureAnalizada)  # Si la desviaciones estandar 0, es una variable estatica y no necesitamos tramificarla

            if desv_std_feature.sum() != 0:
                modelo_discretizador = KBinsDiscretizer(n_bins=numTramos, encode='ordinal', strategy="quantile").fit(featureAnalizada)
                featureTramificada = pd.DataFrame(data=modelo_discretizador.fit_transform(featureAnalizada), index=featuresFichero.index, columns=featureAnalizada.columns)

                # Analisis del desbalanceo positivos/negativos dentro de cada feature
                featureTramificadaConTarget = pd.concat([featureTramificada, targetsFichero], axis=1)
                featureTramificadaConTarget['TARGET'] = np.where(featureTramificadaConTarget['TARGET']==True, 1, 0)
                analisisBalanceoTramos = featureTramificadaConTarget.groupby(by=[cabecera[indice]])["TARGET"]
                num_todos = featureTramificadaConTarget.shape[0]
                tramo_estadisticas = analisisBalanceoTramos.aggregate(np.sum).to_frame().rename(columns={'TARGET':'positivos_en_tramo'}, inplace=False)
                tramo_estadisticas['totales_en_tramo'] = analisisBalanceoTramos.count()
                tramo_estadisticas['pct_positivos_en_tramo'] = 100 * tramo_estadisticas['positivos_en_tramo'] / tramo_estadisticas['totales_en_tramo']
                num_positivos_totales = tramo_estadisticas['positivos_en_tramo'].aggregate(np.sum)
                #print("Feature = '" + cabecera[indice] + "' (indice: " + str(indice) + ") - Tiene " + str(num_positivos_totales) + " positivos (" + "{:.2f}".format(round(100*num_positivos_totales/num_todos, 2)) + "% de "+str(num_todos)+")" + ". Cada tramo debe tener suficientes positivos (>5%):" )
                tramo_estadisticas = tramo_estadisticas.drop(columns=['totales_en_tramo', 'positivos_en_tramo'])
                # pd.set_option('display.max_columns', 30);  print(tramo_estadisticas)

                calidad_discretizacion_stddev = pd.Series.std(tramo_estadisticas['pct_positivos_en_tramo'])
                print("Tramificacion - CALIDAD - Feature = '" + cabecera[indice] + "' -> STD_DEV del porcentaje de positivos en tramo = " + str(calidad_discretizacion_stddev) + " (lo ideal es que sea 0: tramos con igual balanceo de positivos)")
                tramif_calidad_stddev_balanceotramo = tramif_calidad_stddev_balanceotramo.append({'STD_DEV_del_pct_positivos_en_tramo':calidad_discretizacion_stddev, 'FEATURE':cabecera[indice]}, ignore_index = True)

                #TODO Podríamos ordenar los tramos según su 'pct_positivos_en_tramo' y reasignar ese valor "orden" a la featureTramificada. PERO ES COMPLICADO GUARDARLO PARA USARLO EN EL MODO FUTURO. Por tanto, de momento no lo hago
                #tramo_estadisticas = tramo_estadisticas.sort_values(by='pct_positivos_en_tramo', ascending=False)
                #mapeoTramoOrdenado = tramo_estadisticas.index

                # Finalmente, damos el cambiazo a la feature en el DataFrame
                featuresFichero[cabecera[indice]] = featureTramificada

                #Y guardamos el tramificador DE ESTA FEATURE
                # print("Guardando modelo tramificador de feature: " + path_tramificador_feature)
                pickle.dump(modelo_discretizador, open(path_tramificador_feature, 'wb'))

                del featureAnalizada; del featureTramificada; del modelo_discretizador # Fin del IF
        # Fin del WHILE

        print("CALIDAD DE LA TRAMIFICACION - STD_DEV del porcentaje de positivos en tramo (lo ideal es que sea 0: tramos con igual balanceo de positivos):")
        pd.set_option('display.max_columns', 30);
        pd.set_option('display.max_rows', tramif_calidad_stddev_balanceotramo.shape[0] + 1)
        print(tramif_calidad_stddev_balanceotramo.to_string(index=False))
        tramif_calidad_stddev_media = pd.Series.mean(tramif_calidad_stddev_balanceotramo['STD_DEV_del_pct_positivos_en_tramo'])
        print("CALIDAD DE LA TRAMIFICACION (ideal es 0): " +str(tramif_calidad_stddev_media))


        if modoDebug and modoTiempo == "pasado":
            print("FUNCIONES DE DENSIDAD (tramificadas):")
            for column in featuresFichero:
                path_dibujo = dir_subgrupo_img + column + "_TRAMIF.png"
                print("Guardando distrib de col tramificada: " + column + " en fichero: " + path_dibujo)
                datos_columna = featuresFichero[column]
                sns.distplot(datos_columna, kde=False, color='red', bins=dibujoBins)
                plt.title(column + " (TRAMIF)", fontsize=10)
                plt.savefig(path_dibujo, bbox_inches='tight')
                plt.clf(); plt.cla(); plt.close()  # Limpiando dibujo

        del indice; del cabecera; del numTramos  # fin del IF pasado

    if modoTiempo == "futuro":

        while indice < (len(cabecera) - 1):
            indice = indice + 1
            # print("Tramificando feature '" + cabecera[indice] + "' (indice: " + str(indice) + ")")
            featureAnalizada = featuresFichero[cabecera[indice]]  # Feature analizada
            featureAnalizada = featureAnalizada.to_frame()  # Conversion a dataframe
            featureAnalizada.set_index(featuresFichero.index)

            path_tramificador_feature = path_modelo_tramificador+"_FEAT_"+cabecera[indice]

            if os.path.exists(path_tramificador_feature):  # Si existe tramificador DE ESTA FEATURE, lo usamos. Si no, informamos en el log
                modelo_discretizador = pickle.load(open(path_tramificador_feature, 'rb'))
                featureTramificada = pd.DataFrame(data=modelo_discretizador.transform(featureAnalizada),
                                                  index=featuresFichero.index, columns=featureAnalizada.columns)
                del modelo_discretizador

            else:
                print("No hay tramificador guardado (del pasado) de esta feature: ", path_tramificador_feature)
                featureTramificada = featureAnalizada  # DEFAULT por si el modelo tramificador no se creo (ej. si la variable era estatica)

            # Finalmente, damos el cambiazo a la feature en el DataFrame
            featuresFichero[cabecera[indice]] = featureTramificada

        del indice; del cabecera; del numTramos  # Fin del IF

    return featuresFichero


def normalizarFeatures(featuresFichero, path_modelo_normalizador, dir_subgrupo_img, modoTiempo, path_indices_out_capa5, pathCsvIntermedio):
  print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " ----- normalizarFeatures ------")
  print("NORMALIZACION: hacemos que todas las features tengan distribución gaussiana media 0 y varianza 1. El target no se toca.")
  print("PARAMS --> " + path_modelo_normalizador + "|" + modoTiempo + "|" + str(modoDebug))
  print("featuresFichero: " + str(featuresFichero.shape[0]) + " x " + str(featuresFichero.shape[1]))
  print("path_modelo_normalizador: " + path_modelo_normalizador)
  print("pathCsvIntermedio: " + pathCsvIntermedio)

  ################################### NORMALIZACIÓN BoX-COX ####################################################
  # # Los features deben ser estrictamente positivos
  # featuresModificable=featuresFichero
  # for nombreColumna in featuresModificable.columns:
  #     minim = min(featuresModificable[nombreColumna])
  #     featuresModificable[nombreColumna] = featuresModificable[nombreColumna] - minim + 1
  #
  # if modoTiempo == "pasado":
  #     modelo_normalizador = PowerTransformer(method='box-cox', standardize=True, copy=True).fit(featuresModificable)
  #     pickle.dump(modelo_normalizador, open(path_modelo_normalizador, 'wb'))
  #
  # # Pasado o futuro: Cargar normalizador
  # modelo_normalizador = pickle.load(open(path_modelo_normalizador, 'rb'))
  #
  # print("Aplicando normalizacion, manteniendo indices y nombres de columnas...")
  # featuresFicheroNorm = pd.DataFrame(data=modelo_normalizador.transform(featuresModificable), index=featuresModificable.index,
  #                                    columns=featuresModificable.columns)
  #######################################################################################

  ################################### NORMALIZACIÓN YEO-JOHNSON ####################################################
  print("Normalizando cada feature...")
  # Vamos a normalizar z-score (media 0, std_dvt=1), pero yeo-johnson tiene un bug (https://github.com/scipy/scipy/issues/10821) que se soluciona sumando una constante a toda la matriz, lo cual no afecta a la matriz normalizada
  featuresFichero = featuresFichero + 1.015815

  if modoTiempo == "pasado":
      # Con el "normalizador COMPLEJO" solucionamos este bug: https://github.com/scikit-learn/scikit-learn/issues/14959  --> Aplicar los cambios indicados a:_/home/carloslinux/Desktop/PROGRAMAS/anaconda3/envs/BolsaPython/lib/python3.7/site-packages/sklearn/preprocessing/_data.py
      modelo_normalizador = make_pipeline(StandardScaler(with_std=False), PowerTransformer(method='yeo-johnson', standardize=True, copy=True), ).fit(featuresFichero) #COMPLEJO
      #modelo_normalizador = PowerTransformer(method='yeo-johnson', standardize=True, copy=True).fit(featuresFichero)
      pickle.dump(modelo_normalizador, open(path_modelo_normalizador, 'wb'))


  # Pasado o futuro: Cargar normalizador
  modelo_normalizador = pickle.load(open(path_modelo_normalizador, 'rb'))

  print("Aplicando normalizacion, manteniendo indices y nombres de columnas...")
  featuresFicheroNorm = pd.DataFrame(data=modelo_normalizador.transform(featuresFichero), index=featuresFichero.index, columns=featuresFichero.columns)

  print("featuresFicheroNorm:" + str(featuresFicheroNorm.shape[0]) + " x " + str(featuresFicheroNorm.shape[1]))
  featuresFicheroNorm.to_csv(pathCsvIntermedio + ".normalizado.csv", index=True, sep='|')  # UTIL para testIntegracion

  if modoDebug and modoTiempo == "pasado":
    print("FUNCIONES DE DENSIDAD (normalizadas):")
    for column in featuresFicheroNorm:
        path_dibujo = dir_subgrupo_img + column + "_NORM.png"
        print("Guardando distrib de col normalizada: " + column + " en fichero: " + path_dibujo)
        datos_columna = featuresFicheroNorm[column]
        sns.distplot(datos_columna, kde=False, color='red', bins=dibujoBins)
        plt.title(column+" (NORM)", fontsize=10)
        plt.savefig(path_dibujo, bbox_inches='tight')
        plt.clf(); plt.cla(); plt.close()  # Limpiando dibujo

  return featuresFicheroNorm


def comprobarSuficientesClasesTarget(featuresFicheroNorm, targetsFichero):
  print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " ----- comprobarSuficientesClasesTarget ------")
  print("featuresFicheroNorm: " + str(featuresFicheroNorm.shape[0]) + " x " + str(featuresFicheroNorm.shape[1]) +"  Y  " + "targetsFichero: " + str(targetsFichero.shape[0]) + " x " + str(targetsFichero.shape[1]))

  y_unicos = np.unique(targetsFichero)
  # print("Clases encontradas en el target: ")
  # print(y_unicos)
  return y_unicos.size


def reducirFeaturesYGuardar(path_modelo_reductor_features, path_modelo_pca, path_pesos_pca, featuresFicheroNorm, targetsFichero, pathCsvReducido, pathCsvFeaturesElegidas, varianzaAcumuladaDeseada, dir_subgrupo_img, modoTiempo, maxFeatReducidas):
  print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " ----- reducirFeaturesYGuardar ------")
  print("path_modelo_reductor_features --> " + path_modelo_reductor_features)
  print("path_modelo_pca --> " + path_modelo_pca)
  print("path_pesos_pca --> " + path_pesos_pca)
  print("featuresFicheroNorm: " + str(featuresFicheroNorm.shape[0]) + " x " + str(featuresFicheroNorm.shape[1]))
  print("targetsFichero: " + str(targetsFichero.shape[0]) + " x " + str(targetsFichero.shape[1]))
  print("pathCsvReducido --> " + pathCsvReducido)
  print("pathCsvFeaturesElegidas --> " + pathCsvFeaturesElegidas)
  print("varianzaAcumuladaDeseada (PCA) --> " + str(varianzaAcumuladaDeseada))
  print("dir_subgrupo_img --> " + dir_subgrupo_img)
  print("modoTiempo: " + modoTiempo)
  print("maxFeatReducidas: " + maxFeatReducidas)

  print("**** REDUCCION DE DIMENSIONES*****")

  print("** Recursive Feature Elimination (RFE) (que se parece a la técnica Step-wise) **")
  # Create the RFE object and compute a cross-validated score.

  # Comparación de clasificadores
  print('CLASIFICADORES - DENTRO DEL HILO DE EJECUCIÓN')

  if modoTiempo == "pasado":
    # OPCIÓN NO USADA: Si quisiera probar varios clasificadores
    probarVariosClasificadores=False

    estimador_interno = AdaBoostClassifier(n_estimators=50, learning_rate=0.3)
    #estimador_interno = SVC(C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
    #estimador_interno = SVR(kernel='linear', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
    # estimador_interno = RandomForestClassifier(max_depth=4, n_estimators=60, criterion="gini", min_samples_split=2, min_samples_leaf=1,
    #                                    min_weight_fraction_leaf=0., max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.,
    #                                    min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-1, random_state=None,
    #                                    verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)

    # accuracy,balanced_accuracy,average_precision,neg_brier_score,f1,f1_micro,f1_macro,f1_weighted,roc_auc,roc_auc_ovr,roc_auc_ovo,roc_auc_ovr_weighted,roc_auc_ovo_weighted
    #Es mejor roc_auc que f1 y que average_precision/precision. El roc_auc_ovo_weighted no mejora, y roc_auc_ovr_weighted es peor.
    rfecv_scoring = 'roc_auc'

    if probarVariosClasificadores:
        print('Se analiza el accuracy de varios tipos de clasificadores...')
        classifiers = [
        SVC(kernel="rbf"),
        AdaBoostClassifier(n_estimators=50, learning_rate=1.),
        RandomForestClassifier(n_estimators=60, criterion="entropy", max_depth=500, min_samples_split=1, min_samples_leaf=2, min_weight_fraction_leaf=0.,
                            max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0., min_impurity_split=None,
                            bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False,
                            class_weight=None, ccp_alpha=0.0, max_samples=None)]

        scoreAnterior = 0
        numFeaturesAnterior = 9999
        for clasificadorActual in classifiers:
            rfecv = RFECV(estimator=clasificadorActual, step=1, min_features_to_select=7, cv=StratifiedKFold(n_splits=cv_todos, shuffle=True), scoring=rfecv_scoring, verbose=0, n_jobs=-1)
            rfecv.fit(featuresFicheroNorm, targetsFichero)
            print('Accuracy del clasificadorActual: {:.2f}'.format(rfecv.score(featuresFicheroNorm, targetsFichero)))
            print("Optimal number of features : %d" % rfecv.n_features_)

            if scoreAnterior < rfecv.score(featuresFicheroNorm, targetsFichero):
                if numFeaturesAnterior > rfecv.n_features_:
                    print("Se cambia el clasificador elegido")
                    estimador_interno = clasificadorActual
                    scoreAnterior = rfecv.score(featuresFicheroNorm, targetsFichero)
                    numFeaturesAnterior = rfecv.n_features_

    # The "accuracy" scoring is proportional to the number of correct classifications
    num_filas_en_cada_trozo = targetsFichero.shape[0] / cv_todos
    if num_filas_en_cada_trozo < 10:  # La funcion fit() de RFECV exige que haya al menos 10 muestras en el vector target
        n_splits_corregido = math.floor((cv_todos / 10) * num_filas_en_cada_trozo)
    else:
        n_splits_corregido = cv_todos

    print("n_splits_corregido -->" + str(n_splits_corregido))
    rfecv_modelo = RFECV(estimator=estimador_interno, step=rfecv_step, min_features_to_select=4, cv=StratifiedKFold(n_splits=n_splits_corregido, shuffle=True), scoring=rfecv_scoring, verbose=0, n_jobs=-1)
    print("rfecv_modelo -> fit ...")
    targetsLista = targetsFichero["TARGET"].tolist()
    rfecv_modelo.fit(featuresFicheroNorm, targetsLista)
    print("rfecv_modelo -> dump ...")
    pickle.dump(rfecv_modelo, open(path_modelo_reductor_features, 'wb'))


  ################ REDUCCION DE FEATURES para pasado y futuro: ###########
  rfecv_modelo = pickle.load(open(path_modelo_reductor_features, 'rb'))
  print("Numero original de features: %d" % featuresFicheroNorm.shape[1])
  print("Numero optimo de features: %d" % rfecv_modelo.n_features_)

  if rfecv_modelo.n_features_ > int(maxFeatReducidas):
      print("El reductor de dimensiones no es capaz de reducir a un numero razonable/manejable de dimensiones (pocas). Por tanto, no seguimos calculando nada para este subgrupo.")

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
          plt.clf(); plt.cla(); plt.close()  # Limpiando dibujo

      columnas = featuresFicheroNorm.columns
      numColumnas = columnas.shape[0]
      columnasSeleccionadas =[]
      for i in range(numColumnas):
          if(rfecv_modelo.support_[i] == True):
              columnasSeleccionadas.append(columnas[i])


      # print("Mascara de features seleccionadas (rfecv_modelo.support_):")
      # print(rfecv_modelo.support_)
      # print("El ranking de importancia de las features (rfecv_modelo.ranking_) no distingue las features mas importantes dentro de las seleccionadas:")
      # print(rfecv_modelo.ranking_)

      featuresFicheroNormElegidas = featuresFicheroNorm[columnasSeleccionadas]
      print("Features seleccionadas escritas en: " + pathCsvFeaturesElegidas)
      featuresFicheroNormElegidas.head(1).to_csv(pathCsvFeaturesElegidas, index=False, sep='|')

      ########### PCA: base de funciones ortogonales (con combinaciones de features) ########
      if True:
          print("** PCA (Principal Components Algorithm) **")

          if modoTiempo == "pasado":
              print("Usando PCA, creamos una NUEVA BASE DE FEATURES ORTOGONALES y cogemos las que tengan un impacto agregado sobre el X% de la varianza del target. Descartamos el resto.")
              #modelo_pca_subgrupo = PCA(n_components=varianzaAcumuladaDeseada, svd_solver='full')  # Variaza acumulada sobre el target
              modelo_pca_subgrupo = PCA(n_components='mle', svd_solver='full')  # Metodo "MLE de Minka": https://vismod.media.mit.edu/tech-reports/TR-514.pdf
              # modelo_pca_subgrupo = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0,
              #                            n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07,
              #                            metric='euclidean', init='random', verbose=0, random_state=None,
              #                            method='barnes_hut', angle=0.5,
              #                            n_jobs=-1)  # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
              print(modelo_pca_subgrupo)
              featuresFicheroNorm_pca = modelo_pca_subgrupo.fit_transform(featuresFicheroNormElegidas)
              print("modelo_pca_subgrupo -> dump ...")
              pickle.dump(modelo_pca_subgrupo, open(path_modelo_pca, 'wb'))
          else:
              print("modelo_pca_subgrupo -> load ...")
              modelo_pca_subgrupo = pickle.load(open(path_modelo_pca, 'rb'))
              print(modelo_pca_subgrupo)
              featuresFicheroNorm_pca = modelo_pca_subgrupo.transform(featuresFicheroNormElegidas)

          print("Dimensiones del dataframe tras PCA: " + str(featuresFicheroNorm_pca.shape[0]) + " x " + str(featuresFicheroNorm_pca.shape[1]))

          print("Las features están ya normalizadas, reducidas y en base ortogonal PCA. DESCRIBIMOS lo que hemos hecho y GUARDAMOS el dataset.")
          num_columnas_pca = featuresFicheroNorm_pca.shape[1]
          columnas_pca = ["pca_" + f"{i:0>2}" for i in range(num_columnas_pca)]  # Hacemos left padding con la funcion f-strings
          featuresFicheroNorm_pca_df = DataFrame(featuresFicheroNorm_pca, columns=columnas_pca, index=featuresFicheroNorm.index)
          print(featuresFicheroNorm_pca_df.head())
          featuresFicheroNormElegidas = featuresFicheroNorm_pca_df

          print("Matriz de pesos de las features en la base de funciones PCA: " + path_pesos_pca)
          pcaMatriz = pd.DataFrame(modelo_pca_subgrupo.components_)
          pcaMatriz.columns = columnasSeleccionadas
          columnas_pca_df = pd.DataFrame(columnas_pca)
          pcaMatriz = pd.concat([columnas_pca_df, pcaMatriz], axis=1)
          pcaMatriz.to_csv(path_pesos_pca, index=False, sep='|')


      ### Guardar a fichero
      # print("Muestro las features + targets antes de juntarlas...")
      # print("FEATURES (sample):")
      # print(featuresFicheroNormElegidas.head())
      print("featuresFicheroNormElegidas: " + str(featuresFicheroNormElegidas.shape[0]) + " x " + str(featuresFicheroNormElegidas.shape[1]))
      # print("TARGETS (sample):")
      # print(targetsFichero.head())

      featuresytargets = pd.concat([featuresFicheroNormElegidas.reset_index(drop=True), targetsFichero.reset_index(drop=True)], axis=1)  #Column bind
      featuresytargets.set_index(featuresFicheroNormElegidas.index, inplace=True)
      # print("FEATURES+TARGETS juntas (sample):")
      # print(featuresytargets.head())
      print("Justo antes de guardar, featuresytargets: " + str(featuresytargets.shape[0]) + " x " + str(featuresytargets.shape[1]))
      featuresytargets.to_csv(pathCsvReducido, index=True, sep='|')


################## MAIN ###########################################################

if pathCsvCompleto.endswith('.csv') and os.path.isfile(pathCsvCompleto) and os.stat(pathCsvCompleto).st_size > 0:

    featuresFichero1, targetsFichero = leerFeaturesyTarget(pathCsvCompleto, dir_subgrupo_img, compatibleParaMuchasEmpresas, pathModeloOutliers, modoTiempo, maxFilasEntrada)

    # NORMALIZAR Y TRAMIFICAR
    # featuresFichero2 = normalizarFeatures(featuresFichero1, path_modelo_normalizador, dir_subgrupo_img, modoTiempo, path_indices_out_capa5, pathCsvIntermedio)
    #featuresFichero3 = tramificarFeatures(numTramos, featuresFichero2, targetsFichero, path_modelo_tramificador, dir_subgrupo_img, modoTiempo)

    # NORMALIZAR, PERO SIN TRAMIFICAR: leer apartado 4.3 de https://eprints.ucm.es/56355/1/TFM_MPP_Jul19%20%281%29Palau.pdf
    featuresFichero2 = normalizarFeatures(featuresFichero1, path_modelo_normalizador, dir_subgrupo_img, modoTiempo, path_indices_out_capa5, pathCsvIntermedio)
    featuresFichero3 = featuresFichero2

    # NO NORMALIZAR y NO TRAMIFICAR
    # featuresFichero2 = featuresFichero1
    # featuresFichero3 = featuresFichero2

    #-----  Comprobar las clases del target:
    numclases = comprobarSuficientesClasesTarget(featuresFichero3, targetsFichero)

    if(modoTiempo == "pasado" and numclases <= 1):
        print("El subgrupo solo tiene " + str(numclases) + " clases en el target. Abortamos...")
    else:
        reducirFeaturesYGuardar(path_modelo_reductor_features, path_modelo_pca, path_pesos_pca, featuresFichero3, targetsFichero, pathCsvReducido, pathCsvFeaturesElegidas, varianza, dir_subgrupo_img, modoTiempo, maxFeatReducidas)


print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " ------------ FIN de capa 5 ----------------")

