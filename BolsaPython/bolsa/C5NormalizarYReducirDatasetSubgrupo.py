import sys
import os
import pandas as pd
from pathlib import Path
from random import sample, choice

from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
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


print("**** CAPA 5  --> Selección de variables/ Reducción de dimensiones (para cada subgrupo) ****")
print("URL PCA: https://scikit-learn.org/stable/modules/unsupervised_reduction.html")
print("URL Feature Selection: https://scikit-learn.org/stable/modules/feature_selection.html")

##################################################################################################
print("PARAMETROS: ")
dir_subgrupo = sys.argv[1]
modoTiempo = sys.argv[2]
maxFeatReducidas = sys.argv[3]
print("dir_subgrupo = %s" % dir_subgrupo)
print("modoTiempo = %s" % modoTiempo)
print("maxFeatReducidas = %s" % maxFeatReducidas)

varianza=0.90
compatibleParaMuchasEmpresas = True  # Si hay muchas empresas, debo hacer ya el undersampling (en vez de capa 6)
modoDebug = False  # En modo debug se pintan los dibujos. En otro caso, se evita calculo innecesario
pathCsvCompleto = dir_subgrupo + "COMPLETO.csv"
dir_subgrupo_img = dir_subgrupo + "img/"
pathCsvIntermedio = dir_subgrupo + "intermedio.csv"
pathCsvReducido = dir_subgrupo + "REDUCIDO.csv"
pathModeloOutliers = (dir_subgrupo + "DETECTOR_OUTLIERS.tool").replace("futuro", "pasado") # Siempre lo cojo del pasado
path_modelo_normalizador = (dir_subgrupo + "NORMALIZADOR.tool").replace("futuro", "pasado") # Siempre lo cojo del pasado
path_indices_out_capa5 = (dir_subgrupo + "indices_out_capa5.indices")
path_modelo_reductor_features = (dir_subgrupo + "REDUCTOR.tool").replace("futuro", "pasado") # Siempre lo cojo del pasado
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

def leerFeaturesyTarget(path_csv_completo, path_dir_img, compatibleParaMuchasEmpresas, pathModeloOutliers, modoTiempo, modoDebug):
  print("----- leerFeaturesyTarget ------")
  print("PARAMS --> " + path_csv_completo + "|" + path_dir_img + "|" + str(compatibleParaMuchasEmpresas) + "|" + pathModeloOutliers + "|" + modoTiempo + "|" + str(modoDebug))

  print("Cargar datos (CSV)...")
  entradaFeaturesYTarget = pd.read_csv(filepath_or_buffer=path_csv_completo, sep='|')
  num_nulos_por_fila_1 = np.logical_not(entradaFeaturesYTarget.isnull()).sum()
  print("entradaFeaturesYTarget: " + str(entradaFeaturesYTarget.shape[0]) + " x " + str(entradaFeaturesYTarget.shape[1]))
  indiceFilasFuturasTransformadas1 = entradaFeaturesYTarget.index.values  # DEFAULT
  indiceFilasFuturasTransformadas2 = entradaFeaturesYTarget.index.values  # DEFAULT

  ################# Borrado de columnas nulas enteras ##########
  print("MISSING VALUES (COLUMNAS) - Borramos las columnas (features) que sean siempre NaN...")
  entradaFeaturesYTarget2 = entradaFeaturesYTarget.dropna(axis=1, how='all') #Borrar COLUMNA si TODOS sus valores tienen NaN
  print("entradaFeaturesYTarget2 (columnas nulas borradas): " + str(entradaFeaturesYTarget2.shape[0]) + " x " + str(entradaFeaturesYTarget2.shape[1]))
  # entradaFeaturesYTarget2.to_csv(path_csv_completo + "_TEMP01", index=True, sep='|')  # UTIL ara testIntegracion


  ################# Borrado de filas que tengan algun hueco (tratamiento espacial para el futuro con sus columnas: TARGET y otras) #####
  if modoTiempo == "futuro":
      print("Nos quedamos solo con las velas con antiguedad=0 (futuras)...")
      entradaFeaturesYTarget2 = entradaFeaturesYTarget2[entradaFeaturesYTarget2.antiguedad == 0]
      print("entradaFeaturesYTarget2:" + str(entradaFeaturesYTarget2.shape[0]) + " x " + str(entradaFeaturesYTarget2.shape[1]))
      #print(entradaFeaturesYTarget2.head())


  # print("Pasado o Futuro: Transformacion en la que borro filas. Por tanto, guardo el indice...")
  indiceFilasFuturasTransformadas1 = entradaFeaturesYTarget2.index.values

  print("Borrar columnas especiales (idenficadoras de fila): empresa | antiguedad | mercado | anio | mes | dia | hora | minuto...")
  entradaFeaturesYTarget2 = entradaFeaturesYTarget2.drop('empresa', axis=1).drop('antiguedad', axis=1).drop('mercado', axis=1).drop('anio', axis=1).drop('mes', axis=1).drop('dia', axis=1).drop('hora', axis=1).drop('minuto', axis=1)
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
  # entradaFeaturesYTarget3.to_csv(path_csv_completo + "_TEMP03", index=True, sep='|')  # UTIL ara testIntegracion


  # Limpiar OUTLIERS
  # URL: https://scikit-learn.org/stable/modules/outlier_detection.html
  if modoTiempo == "pasado":
      detector_outliers = IsolationForest(contamination='auto', random_state=42)
      df3aux = entradaFeaturesYTarget3.drop('TARGET', axis=1)
      detector_outliers.fit(df3aux)  # fit 10 trees
      pickle.dump(detector_outliers, open(pathModeloOutliers, 'wb'))
  else:
      df3aux = entradaFeaturesYTarget3  # FUTURO

  # Pasado y futuro:
  print("Cargando modelo detector de outliers: " + pathModeloOutliers)
  detector_outliers = pickle.load(open(pathModeloOutliers, 'rb'))
  flagAnomaliasDf = pd.DataFrame({'marca_anomalia':detector_outliers.predict(df3aux)}) #vale -1 es un outlier; si es un 1, no lo es

  indice3=entradaFeaturesYTarget3.index #lo guardo para pegarlo luego
  entradaFeaturesYTarget3.reset_index(drop=True, inplace=True)
  flagAnomaliasDf.reset_index(drop=True, inplace=True)
  entradaFeaturesYTarget4 = pd.concat([entradaFeaturesYTarget3, flagAnomaliasDf], axis=1)  #Column Bind, manteniendo el índice del DF izquierdo
  entradaFeaturesYTarget4.set_index(indice3, inplace = True) #ponemos el indice que tenia el DF de la izquierda

  entradaFeaturesYTarget4 = entradaFeaturesYTarget4.loc[entradaFeaturesYTarget4['marca_anomalia'] == 1] #Cogemos solo las que no son anomalias
  entradaFeaturesYTarget4 = entradaFeaturesYTarget4.drop('marca_anomalia', axis=1)  #Quitamos la columna auxiliar

  print("entradaFeaturesYTarget4 (sin outliers):" + str(entradaFeaturesYTarget4.shape[0]) + " x " + str(entradaFeaturesYTarget4.shape[1]))
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
      sns.distplot(datos_columna, kde=False, color='red', bins=10)
      plt.title(column, fontsize=10)
      plt.savefig(path_dibujo, bbox_inches='tight')
      #Limpiando dibujo:
      plt.clf()
      plt.cla()
      plt.close()

  return featuresFichero, targetsFichero


def normalizarFeatures(featuresFichero, path_modelo_normalizador, dir_subgrupo_img, modoTiempo, path_indices_out_capa5, pathCsvIntermedio, modoDebug):
  print("----- normalizarFeatures ------")
  print("NORMALIZACION: hacemos que todas las features tengan distribución gaussiana media 0 y varianza 1. El target no se toca.")
  print("PARAMS --> " + path_modelo_normalizador + "|" + modoTiempo + "|" + str(modoDebug))
  print("featuresFichero: " + str(featuresFichero.shape[0]) + " x " + str(featuresFichero.shape[1]))
  print("path_modelo_normalizador: " + path_modelo_normalizador)
  print("pathCsvIntermedio: " + pathCsvIntermedio)

  # Vamos a normalizar z-score (media 0, std_dvt=1), pero yeo-johnson tiene un bug (https://github.com/scipy/scipy/issues/10821) que se soluciona sumando una constante a toda la matriz, lo cual no afecta a la matriz normalizada
  featuresFichero = featuresFichero + 1.015815

  if modoTiempo == "pasado":
      modelo_normalizador = PowerTransformer(method='yeo-johnson', standardize=True, copy=True).fit(featuresFichero)
      pickle.dump(modelo_normalizador, open(path_modelo_normalizador, 'wb'))


  # Pasado o futuro: Cargar normalizador
  modelo_normalizador = pickle.load(open(path_modelo_normalizador, 'rb'))

  print("Aplicando normalizacion, manteniendo indices y nombres de columnas...")
  featuresFicheroNorm = pd.DataFrame(data=modelo_normalizador.transform(featuresFichero), index=featuresFichero.index, columns=featuresFichero.columns)

  print("featuresFicheroNorm:" + str(featuresFicheroNorm.shape[0]) + " x " + str(featuresFicheroNorm.shape[1]))
  # featuresFicheroNorm.to_csv(pathCsvIntermedio + "_TEMP_NORM01", index=True, sep='|')  # UTIL para testIntegracion

  if modoDebug and modoTiempo == "pasado":
    print("FUNCIONES DE DENSIDAD (normalizadas):")
    for column in featuresFicheroNorm:
        path_dibujo = dir_subgrupo_img + column + "_NORM.png"
        print("Guardando distrib de col normalizada: " + column + " en fichero: " + path_dibujo)
        datos_columna = featuresFicheroNorm[column]
        sns.distplot(datos_columna, kde=False, color='red', bins=10)
        plt.title(column+" (NORM)", fontsize=10)
        plt.savefig(path_dibujo, bbox_inches='tight')
        # Limpiando dibujo:
        plt.clf()
        plt.cla()
        plt.close()

  return featuresFicheroNorm


def comprobarSuficientesClasesTarget(featuresFicheroNorm, targetsFichero, modoDebug):
  print("----- comprobarSuficientesClasesTarget ------")
  print("featuresFicheroNorm: " + str(featuresFicheroNorm.shape[0]) + " x " + str(featuresFicheroNorm.shape[1]) +"  Y  " + "targetsFichero: " + str(targetsFichero.shape[0]) + " x " + str(targetsFichero.shape[1]))

  y_unicos = np.unique(targetsFichero)
  # print("Clases encontradas en el target: ")
  # print(y_unicos)
  return y_unicos.size


def reducirFeaturesYGuardar(path_modelo_reductor_features, featuresFicheroNorm, targetsFichero, pathCsvReducido, varianzaAcumuladaDeseada, dir_subgrupo_img, modoTiempo, maxFeatReducidas, modoDebug):
  print("----- reducirFeaturesYGuardar ------")
  print("path_modelo_reductor_features --> " + path_modelo_reductor_features)
  print("featuresFicheroNorm: " + str(featuresFicheroNorm.shape[0]) + " x " + str(featuresFicheroNorm.shape[1]))
  print("targetsFichero: " + str(targetsFichero.shape[0]) + " x " + str(targetsFichero.shape[1]))
  print("pathCsvReducido --> " + pathCsvReducido)
  print("varianzaAcumuladaDeseada (PCA) --> " + str(varianzaAcumuladaDeseada))
  print("dir_subgrupo_img --> " + dir_subgrupo_img)
  print("modoTiempo: " + modoTiempo)
  print("maxFeatReducidas: " + maxFeatReducidas)

  print("**** REDUCCION DE DIMENSIONES*****")

  print("** Recursive Feature Elimination (RFE) (que se parece a la técnica Step-wise) **")
  # Create the RFE object and compute a cross-validated score.

  # Comparación de clasificadores
  print('CLASIFICADORES - DENTRO DEL HILO DE EJECUCIÓN')
  print('Se analiza el accuracy de varios tipos de clasificadores...')

  if modoTiempo == "pasado":
    # OPCIÓN NO USADA: Si quisiera probar varios clasificadores
    probarVariosClasificadores=False

    estimador_interno = AdaBoostClassifier(n_estimators=30, learning_rate=1.)
    # estimador_interno = RandomForestClassifier(max_depth=4, n_estimators=60, criterion="gini", min_samples_split=2, min_samples_leaf=1,
    #                                    min_weight_fraction_leaf=0., max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.,
    #                                    min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-1, random_state=None,
    #                                    verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)

    rfecv_scoring = 'f1'  # accuracy,balanced_accuracy,average_precision,neg_brier_score,f1,f1_micro,f1_macro,f1_weighted,roc_auc,roc_auc_ovr,roc_auc_ovo,roc_auc_ovr_weighted,roc_auc_ovo_weighted

    if probarVariosClasificadores:
        classifiers = [
        SVC(kernel="rbf"),
        AdaBoostClassifier(n_estimators=50, learning_rate=1.),
        RandomForestClassifier(n_estimators=60, criterion="entropy", max_depth=500, min_samples_split=1, min_samples_leaf=2, min_weight_fraction_leaf=0.,
                            max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0., min_impurity_split=None,
                            bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False,
                            class_weight = None, ccp_alpha=0.0, max_samples=None)]

        scoreAnterior = 0
        numFeaturesAnterior = 9999
        for clasificadorActual in classifiers:
            rfecv = RFECV(estimator=clasificadorActual, step=1, min_features_to_select=10, cv=StratifiedKFold(n_splits=8, shuffle=True), scoring=rfecv_scoring, verbose=0, n_jobs=-1)
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
    rfecv_modelo = RFECV(estimator=estimador_interno, step=0.05, min_features_to_select=4, cv=StratifiedKFold(n_splits=12, shuffle=True), scoring=rfecv_scoring, verbose=0, n_jobs=-1)
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

      # print("Mascara de features seleccionadas (rfecv_modelo.support_):")
      # print(rfecv_modelo.support_)
      # print("El ranking de importancia de las features (rfecv_modelo.ranking_) no distingue las features mas importantes dentro de las seleccionadas:")
      # print(rfecv_modelo.ranking_)
      print("Las columnas seleccionadas son:")
      print(columnasSeleccionadas)
      featuresFicheroNormElegidas = featuresFicheroNorm[columnasSeleccionadas]
      # featuresFicheroNormElegidas.to_csv(pathCsvReducido + "_TEMP06", index=False, sep='|')  # UTIL ara testIntegracion

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
      # print("Muestro las features + targets antes de juntarlas...")
      # print("FEATURES (sample):")
      # print(featuresFicheroNormElegidas.head())
      print("featuresFicheroNormElegidas: " + str(featuresFicheroNormElegidas.shape[0]) + " x " + str(featuresFicheroNormElegidas.shape[1]))
      # print("TARGETS (sample):")
      # print(targetsFichero.head())

      featuresytargets = pd.concat([featuresFicheroNormElegidas.reset_index(drop=True), targetsFichero.reset_index(drop=True)], axis=1)  #Column bind
      featuresytargets.set_index(featuresFicheroNormElegidas.index, inplace = True)
      # print("FEATURES+TARGETS juntas (sample):")
      # print(featuresytargets.head())
      print("Justo antes de guardar, featuresytargets: " + str(featuresytargets.shape[0]) + " x " + str(featuresytargets.shape[1]))
      featuresytargets.to_csv(pathCsvReducido, index=True, sep='|')


################## MAIN ###########################################################

if pathCsvCompleto.endswith('.csv') and os.path.isfile(pathCsvCompleto) and os.stat(pathCsvCompleto).st_size > 0:

    featuresFichero, targetsFichero = leerFeaturesyTarget(pathCsvCompleto, dir_subgrupo_img, compatibleParaMuchasEmpresas, pathModeloOutliers, modoTiempo, modoDebug)
    featuresFicheroNorm = normalizarFeatures(featuresFichero, path_modelo_normalizador, dir_subgrupo_img, modoTiempo, path_indices_out_capa5, pathCsvIntermedio, modoDebug)
    numclases = comprobarSuficientesClasesTarget(featuresFicheroNorm, targetsFichero, modoDebug)

    if(modoTiempo == "pasado" and numclases <= 1):
        print("El subgrupo solo tiene " + str(numclases) + " clases en el target. Abortamos...")
    else:
        reducirFeaturesYGuardar(path_modelo_reductor_features, featuresFicheroNorm, targetsFichero, pathCsvReducido, varianza, dir_subgrupo_img, modoTiempo, maxFeatReducidas, modoDebug)


print("------------ FIN de capa 5 ----------------")

