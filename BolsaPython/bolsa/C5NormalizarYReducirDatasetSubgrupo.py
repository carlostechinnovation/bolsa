import sys
import os
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("---- CAPA 5 - Selección de variables/ Reducción de dimensiones (para cada subgrupo) -------")
print("URL: https://scikit-learn.org/stable/modules/unsupervised_reduction.html")
##################################################
print("PARAMETROS: ")
dir_entrada = sys.argv[1]
path_dir_salida = sys.argv[2]
print("dir_entrada = %s" % dir_entrada)
print("path_dir_salida = %s" % path_dir_salida)

######################## FUNCIONES ###########
def normalizarYReducirFeaturesDeFichero(pathEntrada, pathSalida):
  print("Entrada --> " + pathEntrada)
  print("Salida --> " + pathSalida)

  print("Cargar datos (CSV)...")
  entradaFeaturesYTarget = pd.read_csv(filepath_or_buffer=pathEntrada, sep='|')
  print("Mostramos las 5 primeras filas:")
  print(entradaFeaturesYTarget.head())

  #ENTRADA: features (+ target)
  featuresFichero = entradaFeaturesYTarget.drop('TARGET', axis=1)
  #featuresFichero = featuresFichero[1:] #quitamos la cabecera
  targetsFichero = entradaFeaturesYTarget[['TARGET']]

  print("FEATURES (sample):")
  print(featuresFichero.head())
  print("TARGETS (sample):")
  print(targetsFichero.head())

  print("NORMALIZACION: hacemos que todas las features tengan distribución gaussiana media 0 y varianza 1. El target no se toca.")
  featuresFicheroNorm = StandardScaler().fit_transform(featuresFichero)
  print("Features NORMALIZADAS (sample):")
  print(featuresFicheroNorm.head())

  print("REDUCCION DE DIMENSIONES: algoritmo PCA")
  print("Usando PCA, cogemos las features que tengan un impacto agregado sobre el X% de la varianza del target. Descartamos el resto.")
  modelo_pca_subgrupo = PCA(n_components=0.95, svd_solver='full')
  featuresFicheroNormReducidas = modelo_pca_subgrupo.fit_transform(featuresFicheroNorm)

  print("Las features están ya normalizadas y reducidas. DESCRIBIMOS lo que hemos hecho y GUARDAMOS el dataset.")


################## MAIN ########################################
print("Recorremos los CSVs que hay en el DIRECTORIO...")
for entry in os.listdir(dir_entrada):
    path_absoluto_fichero = os.path.join(dir_entrada, entry)
    id_subgrupo = Path(entry).stem
    print("id_subgrupo="+id_subgrupo)

    if os.path.isfile(path_absoluto_fichero):
        pathEntrada = os.path.abspath(entry)
        pathSalida = path_dir_salida +id_subgrupo+ ".csv"
        normalizarYReducirFeaturesDeFichero(path_absoluto_fichero, pathSalida)



############################################################
