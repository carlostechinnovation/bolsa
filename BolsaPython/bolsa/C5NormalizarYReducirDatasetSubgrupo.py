import sys
import os
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn import metrics
import numpy as np
from sklearn import linear_model

print("---- CAPA 5 - Selección de variables/ Reducción de dimensiones (para cada subgrupo) -------")
print("URL PCA: https://scikit-learn.org/stable/modules/unsupervised_reduction.html")
print("URL Feature Selection: https://scikit-learn.org/stable/modules/feature_selection.html")
##################################################
print("PARAMETROS: ")
dir_entrada = sys.argv[1]
path_dir_salida = sys.argv[2]
varianza=0.90
print("dir_entrada = %s" % dir_entrada)
print("path_dir_salida = %s" % path_dir_salida)

######################## FUNCIONES ###########
def normalizarYReducirFeaturesDeFichero(pathEntrada, pathSalida, varianzaAcumuladaDeseada):
  print("Entrada --> " + pathEntrada)
  print("Salida --> " + pathSalida)
  print("varianzaAcumuladaDeseada --> " + str(varianzaAcumuladaDeseada))

  print("Cargar datos (CSV)...")
  entradaFeaturesYTarget = pd.read_csv(filepath_or_buffer=pathEntrada, sep='|')

  print("Limpiamos las filas que tengan 1 o mas valores NaN porque son huecos que no deberán estar...")
  entradaFeaturesYTarget2 = entradaFeaturesYTarget.dropna()

  print("Mostramos las 5 primeras filas:")
  print(entradaFeaturesYTarget2.head())

  #ENTRADA: features (+ target)
  featuresFichero = entradaFeaturesYTarget2.drop('TARGET', axis=1)
  #featuresFichero = featuresFichero[1:] #quitamos la cabecera
  targetsFichero = entradaFeaturesYTarget2[['TARGET']]

  print("FEATURES (sample):")
  print(featuresFichero.head())
  print("TARGETS (sample):")
  print(targetsFichero.head())

  print("FUNCIONES DE DENSIDAD:")
  #featuresFichero.plot(kind='density', subplots=True, layout=(8, 8), sharex=False)
  #plt.show()

  print("NORMALIZACION: hacemos que todas las features tengan distribución gaussiana media 0 y varianza 1. El target no se toca.")
  #featuresFicheroNorm = StandardScaler(copy=True, with_mean=True, with_std=True).fit_transform(featuresFichero)
  #featuresFicheroNorm = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True).fit_transform(featuresFichero)
  featuresFicheroNorm = PowerTransformer(method='yeo-johnson').fit_transform(featuresFichero)
  print("Features NORMALIZADAS (sample):")
  print(featuresFicheroNorm)

  print("**** REDUCCION DE DIMENSIONES*****")

  print("** Recursive Feature Elimination (RFE) (que se parece a la técnica Step-wise) **")
  # Create the RFE object and compute a cross-validated score.
  svc_model = SVC(kernel="linear")

  # The "accuracy" scoring is proportional to the number of correct
  # classifications
  rfecv = RFECV(estimator=svc_model, step=1, cv=StratifiedKFold(4), scoring='accuracy')
  rfecv.fit(featuresFicheroNorm, targetsFichero)
  print("Numero original de features: %d" % featuresFichero.shape[1])
  print("Numero optimo de features: %d" % rfecv.n_features_)

  # Plot number of features VS. cross-validation scores
  plt.figure()
  plt.xlabel("Number of features selected")
  plt.ylabel("Cross validation score (nb of correct classifications)")
  plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
  plt.show()

  #print("** PCA (Principal Components Algorithm) **")
  #print("Usando PCA, cogemos las features que tengan un impacto agregado sobre el X% de la varianza del target. Descartamos el resto.")
  #modelo_pca_subgrupo = PCA(n_components=varianzaAcumuladaDeseada, svd_solver='full')
  #print(modelo_pca_subgrupo)
  #featuresFicheroNorm_pca = modelo_pca_subgrupo.fit_transform(featuresFicheroNorm)
  #print(featuresFicheroNorm_pca)
  #print('Dimensiones del dataframe reducido: ' + str(featuresFicheroNorm_pca.shape[0]) + ' x ' + str(featuresFicheroNorm_pca.shape[1]))
  #print("Las features están ya normalizadas y reducidas. DESCRIBIMOS lo que hemos hecho y GUARDAMOS el dataset.")



################## MAIN ########################################
print("Recorremos los CSVs que hay en el DIRECTORIO...")
for entry in os.listdir(dir_entrada):
    path_absoluto_fichero = os.path.join(dir_entrada, entry)
    id_subgrupo = Path(entry).stem
    print("id_subgrupo="+id_subgrupo)

    if os.path.isfile(path_absoluto_fichero):
        pathEntrada = os.path.abspath(entry)
        pathSalida = path_dir_salida +id_subgrupo+ ".csv"
        normalizarYReducirFeaturesDeFichero(path_absoluto_fichero, pathSalida, varianza)



############################################################
print("------------ FIN ----------------")