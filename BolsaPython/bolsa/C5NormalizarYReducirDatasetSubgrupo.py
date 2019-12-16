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
import seaborn as sns


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

  path_dataset_sin_extension = os.path.splitext(pathEntrada)[0]
  print("path_dataset_sin_extension --> " + path_dataset_sin_extension)

  print("Cargar datos (CSV)...")
  entradaFeaturesYTarget = pd.read_csv(filepath_or_buffer=pathEntrada, sep='|')
  num_nulos_por_fila_1 = np.logical_not(entradaFeaturesYTarget.isnull()).sum()

  print("Borramos las columnas (features) que sean siempre NaN. Tambien, limpiamos las filas que tengan 1 o mas valores NaN porque son huecos que no deberán estar...")
  entradaFeaturesYTarget2 = entradaFeaturesYTarget.dropna(axis=1, how='all') #Borrar COLUMNA si TODOS sus valores tienen NaN
  num_nulos_por_fila_2 = np.logical_not(entradaFeaturesYTarget2.isnull()).sum()
  entradaFeaturesYTarget3 = entradaFeaturesYTarget2.dropna(axis=0, how='any')  # Borrar FILA si ALGUNO sus valores tienen NaN
  num_nulos_por_fila_3 = np.logical_not(entradaFeaturesYTarget3.isnull()).sum()

  print("entradaFeaturesYTarget:" + str(entradaFeaturesYTarget.shape[0]) + " x " + str(entradaFeaturesYTarget.shape[1]))
  print("entradaFeaturesYTarget2:" + str(entradaFeaturesYTarget2.shape[0]) + " x " + str(entradaFeaturesYTarget2.shape[1]))
  print("entradaFeaturesYTarget3:" + str(entradaFeaturesYTarget3.shape[0]) + " x " + str(entradaFeaturesYTarget3.shape[1]))

  print("Mostramos las 5 primeras filas:")
  print(entradaFeaturesYTarget3.head())

  #ENTRADA: features (+ target)
  featuresFichero = entradaFeaturesYTarget3.drop('TARGET', axis=1)
  #featuresFichero = featuresFichero[1:] #quitamos la cabecera
  targetsFichero = entradaFeaturesYTarget3[['TARGET']]

  print("FEATURES (sample):")
  print(featuresFichero.head())
  print("TARGETS (sample):")
  print(targetsFichero.head())

  print("FUNCIONES DE DENSIDAD:")
  for column in featuresFichero:
    path_dibujo = path_dataset_sin_extension+"_"+column+".png"
    #print("Pintando columna: " + column + " en fichero: " + path_dibujo)
    #datos_columna = featuresFichero[column]
    #sns.distplot(datos_columna, kde=False, color='red', bins=30)
    #plt.title(column, fontsize=10)
    #plt.savefig(path_dibujo, bbox_inches='tight')


  print("NORMALIZACION: hacemos que todas las features tengan distribución gaussiana media 0 y varianza 1. El target no se toca.")
  #featuresFicheroNorm = StandardScaler(copy=True, with_mean=True, with_std=True).fit_transform(featuresFichero)
  #featuresFicheroNorm = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True).fit_transform(featuresFichero)
  featuresFicheroNorm = PowerTransformer(method='yeo-johnson').fit_transform(featuresFichero)
  print("Features NORMALIZADAS (sample):")
  print(featuresFicheroNorm)
  print("featuresFicheroNorm:" + str(featuresFicheroNorm.shape[0]) + " x " + str(featuresFicheroNorm.shape[1]))


  print("**** REDUCCION DE DIMENSIONES*****")

  print("** Recursive Feature Elimination (RFE) (que se parece a la técnica Step-wise) **")
  # Create the RFE object and compute a cross-validated score.
  svc_model = SVC(kernel="linear")

  # The "accuracy" scoring is proportional to the number of correct
  # classifications
  rfecv = RFECV(estimator=svc_model, step=1, cv=StratifiedKFold(4), scoring='roc_auc') #accuracy
  rfecv.fit(featuresFicheroNorm, targetsFichero)
  print("Numero original de features: %d" % featuresFichero.shape[1])
  print("Numero optimo de features: %d" % rfecv.n_features_)

  # Plot number of features VS. cross-validation scores
  path_dibujo_rfecv = path_dataset_sin_extension + "_RFECV" ".png"
  plt.figure()
  plt.xlabel("Number of features selected")
  plt.ylabel("Cross validation score (nb of correct classifications)")
  plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
  #plt.show()
  #plt.hold(True)
  plt.title("RFECV", fontsize=10)
  plt.savefig(path_dibujo_rfecv, bbox_inches='tight')

  print("** PCA (Principal Components Algorithm) **")
  print("Usando PCA, cogemos las features que tengan un impacto agregado sobre el X% de la varianza del target. Descartamos el resto.")
  modelo_pca_subgrupo = PCA(n_components=varianzaAcumuladaDeseada, svd_solver='full')
  print(modelo_pca_subgrupo)
  featuresFicheroNorm_pca = modelo_pca_subgrupo.fit_transform(featuresFicheroNorm)
  print(featuresFicheroNorm_pca)
  print('Dimensiones del dataframe reducido: ' + str(featuresFicheroNorm_pca.shape[0]) + ' x ' + str(featuresFicheroNorm_pca.shape[1]))
  print("Las features están ya normalizadas y reducidas. DESCRIBIMOS lo que hemos hecho y GUARDAMOS el dataset.")



################## MAIN ########################################
print("Recorremos los CSVs que hay en el DIRECTORIO...")
for entry in os.listdir(dir_entrada):
  path_absoluto_fichero = os.path.join(dir_entrada, entry)

  if (entry.endswith('.csv') and os.path.isfile(path_absoluto_fichero) and os.stat(path_absoluto_fichero).st_size > 0 ):
    print("-------------------------------------------------------------------------------------------------")
    id_subgrupo = Path(entry).stem
    print("id_subgrupo=" + id_subgrupo)
    pathEntrada = os.path.abspath(entry)
    pathSalida = path_dir_salida +id_subgrupo+ ".csv"
    normalizarYReducirFeaturesDeFichero(path_absoluto_fichero, pathSalida, varianza)



############################################################
print("------------ FIN ----------------")