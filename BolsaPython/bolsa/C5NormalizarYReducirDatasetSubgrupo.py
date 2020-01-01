import sys
import os
import pandas as pd
from pathlib import Path
from random import sample, choice

from sklearn.covariance import EllipticEnvelope
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
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
from sklearn.ensemble import IsolationForest, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
import pickle

print("---- CAPA 5 - Selección de variables/ Reducción de dimensiones (para cada subgrupo) -------")
print("URL PCA: https://scikit-learn.org/stable/modules/unsupervised_reduction.html")
print("URL Feature Selection: https://scikit-learn.org/stable/modules/feature_selection.html")

##################################################################################################
print("PARAMETROS: ")
dir_subgrupo = sys.argv[1]
modoTiempo = sys.argv[2]
print("dir_subgrupo = %s" % dir_subgrupo)
print("modoTiempo = %s" % modoTiempo)

varianza=0.90
compatibleParaMuchasEmpresas = True  # Si hay muchas empresas, debo hacer ya el undersampling (en vez de capa 6)
modoDebug = False  # En modo debug se pintan los dibujos. En otro caso, se evita calculo innecesario
pathCsvCompleto = dir_subgrupo + "COMPLETO.csv"
dir_subgrupo_img = dir_subgrupo + "img/"
pathCsvReducido = dir_subgrupo + "REDUCIDO.csv"
pathModeloOutliers = (dir_subgrupo + "DETECTOR_OUTLIERS.tool").replace("futuro", "pasado") # Siempre lo cojo del pasado
path_modelo_normalizador = (dir_subgrupo + "NORMALIZADOR.tool").replace("futuro", "pasado") # Siempre lo cojo del pasado
path_modelo_reductor_features = (dir_subgrupo + "REDUCTOR.tool").replace("futuro", "pasado") # Siempre lo cojo del pasado

print("pathCsvCompleto = %s" % pathCsvCompleto)
print("dir_subgrupo_img = %s" % dir_subgrupo_img)
print("pathCsvReducido = %s" % pathCsvReducido)
print("pathModeloOutliers = %s" % pathModeloOutliers)
print("path_modelo_normalizador = %s" % path_modelo_normalizador)
print("path_modelo_reductor_features = %s" % path_modelo_reductor_features)


######################## FUNCIONES #######################################################

def leerFeaturesyTarget(path_csv_completo, path_dir_img, compatibleParaMuchasEmpresas, pathModeloOutliers, modoTiempo, modoDebug):
  print("----- leerFeaturesyTarget ------")
  print("PARAMS --> " + path_csv_completo + "|" + path_dir_img + "|" + str(compatibleParaMuchasEmpresas) + "|" + pathModeloOutliers + "|" + modoTiempo + "|" + str(modoDebug))

  print("Cargar datos (CSV)...")
  entradaFeaturesYTarget = pd.read_csv(filepath_or_buffer=path_csv_completo, sep='|')
  num_nulos_por_fila_1 = np.logical_not(entradaFeaturesYTarget.isnull()).sum()
  print("entradaFeaturesYTarget:" + str(entradaFeaturesYTarget.shape[0]) + " x " + str(entradaFeaturesYTarget.shape[1]))
  indiceFilasFuturasTransformadas = entradaFeaturesYTarget.index.values

  ################# Borrado de columnas nulas enteras ##########
  print("MISSING VALUES (COLUMNAS) - Borramos las columnas (features) que sean siempre NaN...")
  entradaFeaturesYTarget2 = entradaFeaturesYTarget.dropna(axis=1, how='all') #Borrar COLUMNA si TODOS sus valores tienen NaN
  #num_nulos_por_fila_2 = np.logical_not(entradaFeaturesYTarget2.isnull()).sum()
  print("entradaFeaturesYTarget2 (columnas nulas borradas):" + str(entradaFeaturesYTarget2.shape[0]) + " x " + str(entradaFeaturesYTarget2.shape[1]))


  ################# Borrado de filas que tengan algun hueco (tratamiento espacial para el futuro con sus columnas: TARGET y otras) #####
  if modoTiempo == "futuro":

      print("Nos quedamos solo con las velas con antiguedad=0 (futuras)...")
      entradaFeaturesYTarget2b = entradaFeaturesYTarget2[entradaFeaturesYTarget2.antiguedad == 0]
      print("entradaFeaturesYTarget2b:" + str(entradaFeaturesYTarget2b.shape[0]) + " x " + str(entradaFeaturesYTarget2b.shape[1]))
      print(entradaFeaturesYTarget2b.head())
      print("Transformacion en la que borro o barajo filas. Por tanto, guardo el indice...")
      indiceFilasFuturasTransformadas = entradaFeaturesYTarget2b.index.values

      # Reponer el dataframe 2
      entradaFeaturesYTarget2 = entradaFeaturesYTarget2b

  print("Borrar columnas especiales (informativas): empresa | antiguedad | mercado | anio | mes | dia | hora | minuto...")
  entradaFeaturesYTarget2 = entradaFeaturesYTarget2.drop('empresa', axis=1).drop('antiguedad', axis=1).drop('mercado', axis=1).drop('anio', axis=1).drop('mes', axis=1).drop('dia', axis=1).drop('hora', axis=1).drop('minuto', axis=1)

  print("entradaFeaturesYTarget2:" + str(entradaFeaturesYTarget2.shape[0]) + " x " + str(entradaFeaturesYTarget2.shape[1]))
  print(entradaFeaturesYTarget2.head())


  if modoTiempo == "pasado":
      print("MISSING VALUES (FILAS) - Borramos las FILAS que tengan 1 o mas valores NaN porque son huecos que no deberían estar...")
      entradaFeaturesYTarget3 = entradaFeaturesYTarget2.dropna(axis=0, how='any')  # Borrar FILA si ALGUNO sus valores tienen NaN
      #num_nulos_por_fila_3 = np.logical_not(entradaFeaturesYTarget3.isnull()).sum()
      print("entradaFeaturesYTarget3 (filas algun nulo borradas):" + str(entradaFeaturesYTarget3.shape[0]) + " x " + str(entradaFeaturesYTarget3.shape[1]))

  elif modoTiempo == "futuro":
      print("MISSING VALUES (FILAS) - Para el FUTURO; el target es NULO siempre, pero borramos las filas que tengan ademas otros NULOS...")
      entradaFeaturesYTarget3 = entradaFeaturesYTarget2
      print("entradaFeaturesYTarget3 (filas futuras sin haber borrado nulos):" + str(entradaFeaturesYTarget3.shape[0]) + " x " + str(entradaFeaturesYTarget3.shape[1]))

  print("Mostramos las 5 primeras filas de entradaFeaturesYTarget3:")
  print(entradaFeaturesYTarget3.head())

  # Limpiar OUTLIERS
  # URL: https://scikit-learn.org/stable/modules/outlier_detection.html
  if modoTiempo == "pasado":
      detector_outliers = IsolationForest(contamination='auto', random_state=42)
      detector_outliers.fit(entradaFeaturesYTarget3)  # fit 10 trees
      pickle.dump(detector_outliers, open(pathModeloOutliers, 'wb'))

      # Pasado y futuro:
      print("Cargando modelo detector de outliers: " + pathModeloOutliers)
      detector_outliers = pickle.load(open(pathModeloOutliers, 'rb'))
      outliers_indices = detector_outliers.predict(entradaFeaturesYTarget3) #Si vale -1 es un outlier!!!
      entradaFeaturesYTarget4 = entradaFeaturesYTarget3[np.where(outliers_indices == 1, True, False)]
      print("entradaFeaturesYTarget4 (sin outliers):" + str(entradaFeaturesYTarget4.shape[0]) + " x " + str(entradaFeaturesYTarget4.shape[1]))

  elif modoTiempo == "futuro":  # Para el futuro no podemos quitar OUTLIERS porque la funcion no acepta nulos!!!
      entradaFeaturesYTarget4 = entradaFeaturesYTarget3

  print("Mostramos las 5 primeras filas de entradaFeaturesYTarget4:")
  print(entradaFeaturesYTarget4.head())

  # ENTRADA: features (+ target)

  # Si hay POCAS empresas
  if compatibleParaMuchasEmpresas is False or modoTiempo == "futuro":
    featuresFichero = entradaFeaturesYTarget4.drop('TARGET', axis=1)
    # featuresFichero = featuresFichero[1:] #quitamos la cabecera
    targetsFichero = (entradaFeaturesYTarget4[['TARGET']] == 1)  # Convierto de int a boolean


  # SOLO PARA EL PASADO Si hay MUCHAS empresas (UNDER-SAMPLING para reducir los datos -útil para miles de empresas, pero puede quedar sobreentrenado, si borro casi todas las minoritarias-)
  else:
    print("BALANCEAR los casos positivos y negativos, haciendo downsampling de la clase mayoritaria...")
    print("URL: https://elitedatascience.com/imbalanced-classes")
    ift_minoritaria = entradaFeaturesYTarget4[entradaFeaturesYTarget4.TARGET == True]

    # Se seleccionan tantos target=0 (mayoritaria) como entradas tengo de Target=1 (minoritaria). Se cogeuna muestra de 2xminoritarias, para asegurar que cogemos suficientes valores con target=0. Luego ya nos quedamos sólo con un tamaño mayoriaria=minoritaria
    ift_mayoritaria=pd.DataFrame(index=ift_minoritaria.index.copy(), columns=ift_minoritaria.columns)
    ift_mayoritaria=entradaFeaturesYTarget4.loc[np.random.choice(entradaFeaturesYTarget4.index, 2*ift_minoritaria.shape[0])]
    ift_mayoritaria = ift_mayoritaria[ift_mayoritaria.TARGET == False]
    ift_mayoritaria=ift_mayoritaria.head(ift_minoritaria.shape[0])

    print("ift_mayoritaria:" + str(ift_mayoritaria.shape[0]) + " x " + str(ift_mayoritaria.shape[1]))
    print("ift_minoritaria:" + str(ift_minoritaria.shape[0]) + " x " + str(ift_minoritaria.shape[1]))
    print("Tasa de desbalanceo entre clases = " + str(ift_mayoritaria.shape[0]) + "/" + str(
        ift_minoritaria.shape[0]) + " = " + str(ift_mayoritaria.shape[0] / ift_minoritaria.shape[0]))
    num_muestras_minoria = ift_minoritaria.shape[0]

    print("num_muestras_minoria: " + str(num_muestras_minoria))

    # Juntar ambas clases ya BALANCEADAS. Primero vacío el dataset
    ift_mayoritaria.reset_index(drop=True, inplace=True)
    ift_minoritaria.reset_index(drop=True, inplace=True)

    entradaFeaturesYTarget5=ift_mayoritaria.append(ift_minoritaria)

    print("Las clases ya están balanceadas:")
    print("ift_balanceadas:" + str(entradaFeaturesYTarget5.shape[0]) + " x " + str(entradaFeaturesYTarget5.shape[1]))

    # Se realiza SHUFFLE para mezclar filas con Target True y False, y reseteo los índices
    entradaFeaturesYTarget5=entradaFeaturesYTarget5.sample(frac=1)
    entradaFeaturesYTarget5.reset_index(drop=True, inplace=True)

    featuresFichero = entradaFeaturesYTarget5.drop('TARGET', axis=1)
    targetsFichero = entradaFeaturesYTarget5[['TARGET']]
    targetsFichero=(targetsFichero[['TARGET']] == 1)  # Convierto de int a boolean

  ##################################################

  print("FEATURES (sample):")
  print(featuresFichero.head())
  print("TARGETS (sample):")
  print(targetsFichero.head())

  if modoDebug:
    print("FUNCIONES DE DENSIDAD (sin nulos, pero antes de normalizar):")
    for column in featuresFichero:
      path_dibujo = path_dir_img + column + ".png"
      print("Guardando distrib de col: " + column + " en fichero: " + path_dibujo)
      datos_columna = featuresFichero[column]
      sns.distplot(datos_columna, kde=False, color='red', bins=10)
      plt.title(column, fontsize=10)
      plt.savefig(path_dibujo, bbox_inches='tight')
      #Limpiando dibujo:
      plt.clf()
      plt.cla()
      plt.close()

  return featuresFichero, targetsFichero, indiceFilasFuturasTransformadas


def normalizarFeatures(featuresFichero, path_modelo_normalizador, dir_subgrupo_img, modoTiempo, indiceFilasFuturasTransformadas, modoDebug):
  print("----- normalizarFeatures ------")
  print("NORMALIZACION: hacemos que todas las features tengan distribución gaussiana media 0 y varianza 1. El target no se toca.")
  print("PARAMS --> " + path_modelo_normalizador + "|" + modoTiempo + "|" + str(modoDebug))
  print("featuresFichero:" + str(featuresFichero.shape[0]) + " x " + str(featuresFichero.shape[1]))
  print("path_modelo_normalizador:" + path_modelo_normalizador)
  print("indiceFilasFuturasTransformadas (numero de items):" + str(indiceFilasFuturasTransformadas.shape[0]))

  if modoTiempo == "pasado":
      modelo_normalizador = PowerTransformer(method='yeo-johnson', standardize=True, copy=True).fit(featuresFichero)
      pickle.dump(modelo_normalizador, open(path_modelo_normalizador, 'wb')) # Luego basta cargarlo así --> modelo_normalizador=load(path_modelo_normalizador)
  elif modoTiempo == "futuro":
      modelo_normalizador = pickle.load(open(path_modelo_normalizador, 'rb'))

  print("Aplicando normalizacion...")
  featuresFicheroNorm = modelo_normalizador.transform(featuresFichero)

  print("Metiendo cabeceras...")
  featuresFicheroNorm2 = pd.DataFrame(data=featuresFicheroNorm, columns=featuresFichero.columns)

  print("Features NORMALIZADAS (sample):")
  print(featuresFicheroNorm2)
  print("featuresFicheroNorm2:" + str(featuresFicheroNorm2.shape[0]) + " x " + str(featuresFicheroNorm2.shape[1]))

  if modoDebug:
    print("FUNCIONES DE DENSIDAD (normalizadas):")
    for column in featuresFicheroNorm2:
        path_dibujo = dir_subgrupo_img + column + "_NORM.png"
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


def comprobarSuficientesClasesTarget(featuresFicheroNorm, targetsFichero, modoDebug):
  print("----- comprobarSuficientesClasesTarget ------")
  print("featuresFicheroNorm:" + str(featuresFicheroNorm.shape[0]) + " x " + str(featuresFicheroNorm.shape[1]))
  print("targetsFichero:" + str(targetsFichero.shape[0]) + " x " + str(targetsFichero.shape[1]))

  y_unicos = np.unique(targetsFichero)
  print("Clases encontradas en el target: ")
  print(y_unicos)
  return y_unicos.size


def reducirFeaturesYGuardar(path_modelo_reductor_features, featuresFicheroNorm, targetsFichero, pathCsvReducido, varianzaAcumuladaDeseada, dir_subgrupo_img, modoTiempo, indiceFilasFuturasTransformadas, modoDebug):
  print("----- reducirFeaturesYGuardar ------")
  print("path_modelo_reductor_features --> " + path_modelo_reductor_features)
  print("featuresFicheroNorm:" + str(featuresFicheroNorm.shape[0]) + " x " + str(featuresFicheroNorm.shape[1]))
  print("targetsFichero:" + str(targetsFichero.shape[0]) + " x " + str(targetsFichero.shape[1]))
  print("pathCsvReducido --> " + pathCsvReducido)
  print("varianzaAcumuladaDeseada (PCA) --> " + str(varianzaAcumuladaDeseada))
  print("dir_subgrupo_img --> " + dir_subgrupo_img)
  print("modoTiempo:" + modoTiempo)
  print("indiceFilasFuturasTransformadas (numero de items):" + str(indiceFilasFuturasTransformadas.shape[0]))


  print("**** REDUCCION DE DIMENSIONES*****")

  print("** Recursive Feature Elimination (RFE) (que se parece a la técnica Step-wise) **")
  # Create the RFE object and compute a cross-validated score.

  # Comparación de clasificadores
  print('CLASIFICADORES - DENTRO DEL HILO DE EJECUCIÓN- LUIS')
  print('Se analiza el accuracy de varios tipos de clasificadores...')

  if modoTiempo == "pasado":
    # OPCIÓN NO USADA: Si quisiera probar varios clasificadores
    probarVariosClasificadores=False

    svc_model = AdaBoostClassifier(n_estimators=50, learning_rate=1.)

    if probarVariosClasificadores:
        classifiers = [
        SVC(kernel="linear"),
        AdaBoostClassifier(n_estimators=50, learning_rate=1.),
        RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.,
                            max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0., min_impurity_split=None,
                            bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False,
                            class_weight=None, ccp_alpha=0.0, max_samples=None)]

        scoreAnterior = 0
        numFeaturesAnterior=9999
        for clasificadorActual in classifiers:
            rfecv = RFECV(estimator=clasificadorActual, step=1, min_features_to_select=3, cv=StratifiedKFold(3), scoring='accuracy', verbose=0, n_jobs=8)
            rfecv.fit(featuresFicheroNorm, targetsFichero)
            print('Accuracy del clasificadorActual: {:.2f}'.format(rfecv.score(featuresFicheroNorm, targetsFichero)))
            print("Optimal number of features : %d" % rfecv.n_features_)

            if scoreAnterior < rfecv.score(featuresFicheroNorm, targetsFichero):
                if numFeaturesAnterior > rfecv.n_features_:
                    print("Se cambia el clasificador elegido")
                    svc_model=clasificadorActual
                    scoreAnterior=rfecv.score(featuresFicheroNorm, targetsFichero)
                    numFeaturesAnterior=rfecv.n_features_

    # The "accuracy" scoring is proportional to the number of correct classifications
    rfecv_modelo = RFECV(estimator=svc_model, step=1, min_features_to_select=3, cv=StratifiedKFold(3), scoring='accuracy', verbose=0, n_jobs=8)
    rfecv_modelo.fit(featuresFicheroNorm, targetsFichero)
    pickle.dump(rfecv_modelo, open(path_modelo_reductor_features, 'wb'))


  ################ REDUCCION DE FEATURES para pasado y futuro: ###########
  rfecv_modelo = pickle.load(open(path_modelo_reductor_features, 'rb'))
  print("Numero original de features: %d" % featuresFicheroNorm.shape[1])
  print("Numero optimo de features: %d" % rfecv_modelo.n_features_)

  # Plot number of features VS. cross-validation scores
  if modoDebug:
    path_dibujo_rfecv = dir_subgrupo_img + "SELECCION_VARIABLES_RFECV" ".png"
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

  columnas = featuresFicheroNorm.columns
  numColumnas = columnas.shape[0]
  columnasSeleccionadas =[]
  for i in range(numColumnas):
      if(rfecv_modelo.support_[i] == True):
          columnasSeleccionadas.append(columnas[i])

  print("Mascara de features seleccionadas (rfecv_modelo.support_):")
  print(rfecv_modelo.support_)
  print("El ranking de importancia de las features (rfecv_modelo.ranking_) no distingue las features mas importantes dentro de las seleccionadas:")
  print(rfecv_modelo.ranking_)
  print("Las columnas seleccionadas son:")
  print(columnasSeleccionadas)
  featuresFicheroNormElegidas = featuresFicheroNorm[columnasSeleccionadas]
  print(featuresFicheroNormElegidas)

  ####################### NO LO USAMOS pero lo dejo aqui ########
  #if modoDebug and False:
      #print("** PCA (Principal Components Algorithm) **")
      #print("Usando PCA, cogemos las features que tengan un impacto agregado sobre el X% de la varianza del target. Descartamos el resto.")
      #modelo_pca_subgrupo = PCA(n_components=varianzaAcumuladaDeseada, svd_solver='full')
      #print(modelo_pca_subgrupo)
      #featuresFicheroNorm_pca = modelo_pca_subgrupo.fit_transform(featuresFicheroNorm)
      #print(featuresFicheroNorm_pca)
      #print('Dimensiones del dataframe reducido: ' + str(featuresFicheroNorm_pca.shape[0]) + ' x ' + str(featuresFicheroNorm_pca.shape[1]))
      #print("Las features están ya normalizadas y reducidas. DESCRIBIMOS lo que hemos hecho y GUARDAMOS el dataset.")

  ### Guardar a fichero
  print("Escribiendo las features a CSV...")
  print("Muestro las features + targets antes de juntarlas...")
  print("FEATURES (sample):")
  print(featuresFicheroNormElegidas.head())
  print("TARGETS (sample):")
  print(targetsFichero.head())

  featuresytargets = pd.concat([featuresFicheroNormElegidas.reset_index(drop=True), targetsFichero.reset_index(drop=True)], axis=1)
  featuresytargets.to_csv(pathCsvReducido, index=False, sep='|')

  print("Muestro las features + targets despues de juntarlas...")
  print("FEATURES+TARGETS juntas (sample):")
  print(featuresytargets.head())

  if modoTiempo == "futuro":
      print("Guardamos a CSV los indices de las filas de REDUCIDO respeto a COMPLETO (para cuando haya que reconstruir el final")
      np.savetxt(pathCsvReducido+"_indices", indiceFilasFuturasTransformadas, header="indice", delimiter="|", fmt='%f')



print("################## MAIN ###########################################################")

if pathCsvCompleto.endswith('.csv') and os.path.isfile(pathCsvCompleto) and os.stat(pathCsvCompleto).st_size > 0:

    featuresFichero, targetsFichero, indiceFilasFuturasTransformadas = leerFeaturesyTarget(pathCsvCompleto, dir_subgrupo_img, compatibleParaMuchasEmpresas, pathModeloOutliers, modoTiempo, modoDebug)
    featuresFicheroNorm = normalizarFeatures(featuresFichero, path_modelo_normalizador, dir_subgrupo_img, modoTiempo, indiceFilasFuturasTransformadas, modoDebug)
    numclases = comprobarSuficientesClasesTarget(featuresFicheroNorm, targetsFichero, modoDebug)

    if(modoTiempo == "pasado" and numclases <= 1):
        print("El subgrupo solo tiene " + str(numclases) + " clases en el target. Abortamos...")
    else:
        reducirFeaturesYGuardar(path_modelo_reductor_features, featuresFicheroNorm, targetsFichero, pathCsvReducido, varianza, dir_subgrupo_img, modoTiempo, indiceFilasFuturasTransformadas, modoDebug)

print("------------ FIN de capa 5 ----------------")

