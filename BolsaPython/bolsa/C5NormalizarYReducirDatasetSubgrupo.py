import sys
import os
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV, RFE
from sklearn import metrics
import numpy as np
from sklearn import linear_model
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.externals.joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


print("---- CAPA 5 - Selección de variables/ Reducción de dimensiones (para cada subgrupo) -------")
print("URL PCA: https://scikit-learn.org/stable/modules/unsupervised_reduction.html")
print("URL Feature Selection: https://scikit-learn.org/stable/modules/feature_selection.html")
##################################################
print("PARAMETROS: ")
dir_entrada = sys.argv[1]
path_dir_salida = sys.argv[2]
path_dir_img = sys.argv[3]
varianza=0.90
modoDebug = True  #En modo debug se pintan los dibujos. En otro caso, se evita calculo innecesario
print("dir_entrada = %s" % dir_entrada)
print("path_dir_salida = %s" % path_dir_salida)
print("path_dir_img = %s" % path_dir_img)

######################## FUNCIONES ###########
def leerFeaturesyTarget(pathEntrada, path_dir_img, modoDebug):
  print("----- leerFeaturesyTarget ------")
  print("Entrada --> " + pathEntrada)

  path_dataset_sin_extension = os.path.splitext(pathEntrada)[0]
  id_subgrupo = Path(path_dataset_sin_extension).stem
  print("id_subgrupo=" + id_subgrupo)

  pathModeloOutliers = path_dataset_sin_extension + "_OUTLIERS.model"
  print("pathModeloOutliers --> " + pathModeloOutliers)

  print("Cargar datos (CSV)...")
  entradaFeaturesYTarget = pd.read_csv(filepath_or_buffer=pathEntrada, sep='|')
  num_nulos_por_fila_1 = np.logical_not(entradaFeaturesYTarget.isnull()).sum()

  print("Borramos las columnas (features) que sean siempre NaN. Tambien, limpiamos las filas que tengan 1 o mas valores NaN porque son huecos que no deberán estar...")
  entradaFeaturesYTarget2 = entradaFeaturesYTarget.dropna(axis=1, how='all') #Borrar COLUMNA si TODOS sus valores tienen NaN
  #num_nulos_por_fila_2 = np.logical_not(entradaFeaturesYTarget2.isnull()).sum()
  entradaFeaturesYTarget3 = entradaFeaturesYTarget2.dropna(axis=0, how='any')  # Borrar FILA si ALGUNO sus valores tienen NaN
  #num_nulos_por_fila_3 = np.logical_not(entradaFeaturesYTarget3.isnull()).sum()

  print("entradaFeaturesYTarget:" + str(entradaFeaturesYTarget.shape[0]) + " x " + str(entradaFeaturesYTarget.shape[1]))
  print("entradaFeaturesYTarget2 (columnas nulas borradas):" + str(entradaFeaturesYTarget2.shape[0]) + " x " + str(entradaFeaturesYTarget2.shape[1]))
  print("entradaFeaturesYTarget3 (filas algun nulo borradas):" + str(entradaFeaturesYTarget3.shape[0]) + " x " + str(entradaFeaturesYTarget3.shape[1]))

  # Limpiar OUTLIERS
  # URL: https://scikit-learn.org/stable/modules/outlier_detection.html
  detector_outliers = IsolationForest(n_estimators=10, warm_start=True)
  detector_outliers.fit(entradaFeaturesYTarget3)  # fit 10 trees
  dump(detector_outliers, pathModeloOutliers, compress=True) # Luego basta cargarlo así --> detector_outliers=load(pathModeloOutliers)
  outliers_indices = detector_outliers.predict(entradaFeaturesYTarget3) #Si vale -1 es un outlier!!!
  entradaFeaturesYTarget4 = entradaFeaturesYTarget3[np.where(outliers_indices == 1, True, False)]
  print("entradaFeaturesYTarget4 (sin outliers):" + str(entradaFeaturesYTarget4.shape[0]) + " x " + str(entradaFeaturesYTarget4.shape[1]))


  print("Mostramos las 5 primeras filas:")
  print(entradaFeaturesYTarget4.head())

  #ENTRADA: features (+ target)
  featuresFichero = entradaFeaturesYTarget4.drop('TARGET', axis=1)
  #featuresFichero = featuresFichero[1:] #quitamos la cabecera
  targetsFichero = (entradaFeaturesYTarget4[['TARGET']] == 1)  # Convierto de int a boolean

  print("FEATURES (sample):")
  print(featuresFichero.head())
  print("TARGETS (sample):")
  print(targetsFichero.head())

  if modoDebug:
    print("FUNCIONES DE DENSIDAD (sin nulos, pero antes de normalizar):")
    for column in featuresFichero:
      path_dibujo = path_dir_img + id_subgrupo+"_"+column+".png"
      print("Guardando distrib de col: " + column + " en fichero: " + path_dibujo)
      datos_columna = featuresFichero[column]
      sns.distplot(datos_columna, kde=False, color='red', bins=10)
      plt.title(column, fontsize=10)
      plt.savefig(path_dibujo, bbox_inches='tight')
      #Limpiando dibujo:
      plt.clf()
      plt.cla()
      plt.close()

  return featuresFichero, targetsFichero, path_dataset_sin_extension


def normalizarFeatures(featuresFichero, path_dataset_sin_extension, modoDebug):
  print("----- normalizarFeatures ------")
  print("featuresFichero:" + str(featuresFichero.shape[0]) + " x " + str(featuresFichero.shape[1]))
  print("path_dataset_sin_extension:" + path_dataset_sin_extension)

  path_modelo_normalizador = path_dataset_sin_extension + "_NORM.model"
  print("path_modelo_normalizador:" + path_modelo_normalizador)

  print("NORMALIZACION: hacemos que todas las features tengan distribución gaussiana media 0 y varianza 1. El target no se toca.")
  modelo_normalizador = PowerTransformer(method='yeo-johnson', standardize=True, copy=True).fit(featuresFichero)
  dump(modelo_normalizador, path_modelo_normalizador, compress=True)  # Luego basta cargarlo así --> modelo_normalizador=load(path_modelo_normalizador)

  featuresFicheroNorm = modelo_normalizador.transform(featuresFichero)

  print("Metiendo cabeceras...")
  featuresFicheroNorm2 = pd.DataFrame(data=featuresFicheroNorm, columns=featuresFichero.columns)

  print("Features NORMALIZADAS (sample):")
  print(featuresFicheroNorm2)
  print("featuresFicheroNorm2:" + str(featuresFicheroNorm2.shape[0]) + " x " + str(featuresFicheroNorm2.shape[1]))

  if modoDebug:
    print("FUNCIONES DE DENSIDAD (normalizadas):")
    for column in featuresFicheroNorm2:
        path_dibujo = path_dir_img + id_subgrupo + "_" + column + "_NORM.png"
        print("Guardando distrib de col normalizada: " + column + " en fichero: " + path_dibujo)
        datos_columna = featuresFicheroNorm2[column]
        sns.distplot(datos_columna, kde=False, color='red', bins=10)
        plt.title(column+" (NORM)", fontsize=10)
        plt.savefig(path_dibujo, bbox_inches='tight')
        # Limpiando dibujo:
        plt.clf()
        plt.cla()
        plt.close()

  return featuresFicheroNorm2

def comprobarSuficientesCasos(featuresFicheroNorm, targetsFichero, modoDebug):
  print("----- comprobarSuficientesCasos ------")
  print("featuresFicheroNorm:" + str(featuresFicheroNorm.shape[0]) + " x " + str(featuresFicheroNorm.shape[1]))
  print("targetsFichero:" + str(targetsFichero.shape[0]) + " x " + str(targetsFichero.shape[1]))

  y_unicos = np.unique(targetsFichero)
  print("Clases encontradas en el target: ")
  print(y_unicos)



def reducirFeaturesYGuardar(featuresFicheroNorm, targetsFichero, pathSalidaFeaturesyTargets, varianzaAcumuladaDeseada, path_dataset_sin_extension, modoDebug):
  print("----- reducirFeaturesYGuardar ------")
  print("featuresFicheroNorm:" + str(featuresFicheroNorm.shape[0]) + " x " + str(featuresFicheroNorm.shape[1]))
  print("targetsFichero:" + str(targetsFichero.shape[0]) + " x " + str(targetsFichero.shape[1]))
  print("pathSalidaFeaturesyTargets --> " + pathSalidaFeaturesyTargets)
  print("varianzaAcumuladaDeseada (PCA) --> " + str(varianzaAcumuladaDeseada))
  print("path_dataset_sin_extension --> " + path_dataset_sin_extension)

  print("**** REDUCCION DE DIMENSIONES*****")

  print("** Recursive Feature Elimination (RFE) (que se parece a la técnica Step-wise) **")
  # Create the RFE object and compute a cross-validated score.
  svc_model = SVC(kernel="linear")
  # The "accuracy" scoring is proportional to the number of correct classifications
  rfecv_modelo = RFECV(estimator=svc_model, step=1, min_features_to_select=3, cv=StratifiedKFold(3), scoring='accuracy', verbose=1, n_jobs=8)
  rfecv_modelo.fit(featuresFicheroNorm, targetsFichero)
  print("Numero original de features: %d" % featuresFicheroNorm.shape[1])
  print("Numero optimo de features: %d" % rfecv_modelo.n_features_)
  print("Mascara de features elegidas:")
  print(rfecv_modelo.support_)
  print("Ranking:")
  print(rfecv_modelo.ranking_)

  # Plot number of features VS. cross-validation scores
  if modoDebug:
    path_dibujo_rfecv = path_dataset_sin_extension + "_RFECV" ".png"
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv_modelo.grid_scores_) + 1), rfecv_modelo.grid_scores_)
    plt.title("RFECV", fontsize=10)
    plt.savefig(path_dibujo_rfecv, bbox_inches='tight')
    # Limpiando dibujo:
    plt.clf()
    plt.cla()
    plt.close()

  print("Las features elegidas son (aplicamos la mascara al dataframe de features):")
  columnas = featuresFicheroNorm.columns
  #print(columnas)
  numColumnas = columnas.shape[0]
  print("numColumnas="+str(numColumnas))

  columnasSeleccionadas =[]
  for i in range(numColumnas):
      if(rfecv_modelo.support_[i] == True):
          columnasSeleccionadas.append(columnas[i])

  print("Las columnas seleccionadas son:")
  print(columnasSeleccionadas)
  featuresFicheroNormElegidas = featuresFicheroNorm[columnasSeleccionadas]
  print(featuresFicheroNormElegidas)

  ####################### NO LO USAMOS pero lo dejo aqui ########
  if modoDebug:
      print("** PCA (Principal Components Algorithm) **")
      print("Usando PCA, cogemos las features que tengan un impacto agregado sobre el X% de la varianza del target. Descartamos el resto.")
      modelo_pca_subgrupo = PCA(n_components=varianzaAcumuladaDeseada, svd_solver='full')
      print(modelo_pca_subgrupo)
      featuresFicheroNorm_pca = modelo_pca_subgrupo.fit_transform(featuresFicheroNorm)
      print(featuresFicheroNorm_pca)
      print('Dimensiones del dataframe reducido: ' + str(featuresFicheroNorm_pca.shape[0]) + ' x ' + str(featuresFicheroNorm_pca.shape[1]))
      print("Las features están ya normalizadas y reducidas. DESCRIBIMOS lo que hemos hecho y GUARDAMOS el dataset.")


  ### Guardar a fichero
  print("Escribiendo las features a CSV...")
  featuresytargets = pd.concat([featuresFicheroNormElegidas, targetsFichero], axis=1, join='inner')
  featuresytargets.to_csv(pathSalidaFeaturesyTargets, index=False, sep='|')


################## MAIN ########################################
print("Recorremos los CSVs que hay en el DIRECTORIO...")
for entry in os.listdir(dir_entrada):
  path_absoluto_fichero = os.path.join(dir_entrada, entry)

  if (entry.endswith('.csv') and os.path.isfile(path_absoluto_fichero) and os.stat(path_absoluto_fichero).st_size > 0 ):
    print("-------------------------------------------------------------------------------------------------")
    id_subgrupo = Path(entry).stem
    print("id_subgrupo=" + id_subgrupo)
    pathEntrada = os.path.abspath(entry)
    pathSalidaFeaturesyTargets = path_dir_salida + id_subgrupo + ".csv"
    featuresFichero, targetsFichero, path_dataset_sin_extension = leerFeaturesyTarget(path_absoluto_fichero, path_dir_img, modoDebug)
    featuresFicheroNorm = normalizarFeatures(featuresFichero, path_dataset_sin_extension, modoDebug)
    comprobarSuficientesCasos(featuresFicheroNorm, targetsFichero, modoDebug)
    reducirFeaturesYGuardar(featuresFicheroNorm, targetsFichero, pathSalidaFeaturesyTargets, varianza, path_dataset_sin_extension, modoDebug)


############################################################
print("------------ FIN ----------------")


