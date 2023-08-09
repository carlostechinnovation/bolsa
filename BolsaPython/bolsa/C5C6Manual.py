import datetime
import os.path
import pickle
import sys
import warnings
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from imblearn.combine import SMOTETomek
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.utils import resample
from tabulate import tabulate
from xgboost import XGBClassifier

import C5C6ManualFunciones
import C5C6ManualML

##################################################################################################
# REDEFINICION DE FUNCION PRINT() PARA QUE TENGA TIMESTAMP
old_print = print
def timestamped_print(*args, **kwargs):
  old_print((datetime.datetime.now()).strftime(
    "%Y%m%d_%H%M%S"), *args, **kwargs)
print = timestamped_print
##################################################################################################
def check_file_and_exit(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"El fichero '{file_path}' no existe. Saliendo...")
        exit(-1)

    # Check if the file is empty
    if os.stat(file_path).st_size == 0:
        print(f"El fichero '{file_path}' esta vacio. Saliendo...")
        exit(-1)

    # If the file exists and is not empty, you can proceed with your further logic here
    print(f"El fichero '{file_path}' existe y no esta vacio. La ejecucion continua...")
##################################################################################################

print(" **** CAPA 5  --> Selección de variables/ Reducción de dimensiones (para cada subgrupo) ****")
# print("URL PCA: https://scikit-learn.org/stable/modules/unsupervised_reduction.html")
# print("URL Feature Selection: https://scikit-learn.org/stable/modules/feature_selection.html")

##################################################################################################
print("PARAMETROS: dir_subgrupo modoTiempo maxFeatReducidas maxFilasEntrada desplazamientoAntiguedad")
if len(sys.argv) <= 1:
    print("Falta indicar los PARAMETROS. Saliendo...")
    exit(-1)

# EJEMPLO 1: /bolsa/pasado/subgrupos/SG_2/ pasado 50 20000 0
# EJEMPLO 2: /bolsa/futuro/subgrupos/SG_2/ futuro 50 20000 0

dir_subgrupo = sys.argv[1]
modoTiempo = sys.argv[2]
maxFeatReducidas = sys.argv[3]
maxFilasEntrada = sys.argv[4]
desplazamientoAntiguedad = sys.argv[5]

print("\tdir_subgrupo = %s" % dir_subgrupo)
print("\tmodoTiempo = %s" % modoTiempo)
print("\tmaxFeatReducidas = %s" % maxFeatReducidas)
print("\tmaxFilasEntrada = %s" % maxFilasEntrada)
print("\tdesplazamientoAntiguedad = %s" % desplazamientoAntiguedad)

##################################################################################################
# DEBUG - EMPRESA vigilada en un dia-mes-año
global DEBUG_EMPRESA, DEBUG_ANIO, DEBUG_MES, DEBUG_DIA, DEBUG_FILTRO
DEBUG_EMPRESA = "ADI"
DEBUG_ANIO = 2022
DEBUG_MES = 2
DEBUG_DIA = 17
DEBUG_FILTRO = DEBUG_EMPRESA + "_" + str(DEBUG_ANIO) + "_" + str(DEBUG_MES) + "_" + str(DEBUG_DIA)
##################################################################################################

varianza = 0.93  # Variacion acumulada de las features PCA sobre el target
UMBRAL_VELASMUYANTIGUASELIMINABLES = (
                                             5 * 4.5) * 6  # todas las velas con mas antiguedad que este umbral no se usan para train ni test ni valid. Recom: 90 (4 meses)
UMBRAL_COLUMNAS_DEMASIADOS_NULOS = 0.25  # Porcentaje de nulos en cada columna. Si se supera, borramos toda la columna. Recomendable: 0.40
compatibleParaMuchasEmpresas = False  # Si hay muchas empresas, debo hacer ya el undersampling (en vez de capa 6)
global modoDebug;
modoDebug = False  # VARIABLE GLOBAL: En modo debug se pintan los dibujos. En otro caso, se evita calculo innecesario. Recomendable: False
evitarNormalizarNiTramificar = False  # VARIABLE GLOBAL: si se quiere no normalizar ni tramificar features. Recomendable: False
limpiarOutliers = True
global cv_todos;
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

# Modelos que siempre cogemos del pasado:
pathModeloOutliers = (dir_subgrupo + "DETECTOR_OUTLIERS.tool").replace("futuro", "pasado")
# path_modelo_tramificador = (dir_subgrupo + "tramif/" + "TRAMIFICADOR").replace("futuro", "pasado")
path_modelo_normalizador = (dir_subgrupo + "NORMALIZADOR.tool").replace("futuro", "pasado")
path_indices_out_capa5 = (dir_subgrupo + "indices_out_capa5.indices")
path_modelo_reductor_features = (dir_subgrupo + "REDUCTOR.tool").replace("futuro", "pasado")
path_modelo_pca = (dir_subgrupo + "PCA.tool").replace("futuro", "pasado")
path_pesos_pca = (dir_subgrupo + "PCA_matriz.csv")

# BALANCEO DE CLASES
umbralNecesarioCompensarDesbalanceo = 1  # Umbral de desbalanceo clase positiva/negativa. Si se supera, lo compensamos. Deshabilitado si vale 0
balancearConSmoteSoloTrain = True  # SMOTE sobre minoritaria y mayoritaria. Sólo aplicable a pasado-Train; no a test ni validacion ni a futuro.
balancearUsandoDownsampling = False  # Downsampling de clase mayoritaria. Sólo aplicable a pasado-Train; no a test ni validacion ni a futuro.

print("\tpathCsvCompleto = %s" % pathCsvCompleto)
print("\tdir_subgrupo_img = %s" % dir_subgrupo_img)
print("\tpathCsvIntermedio = %s" % pathCsvIntermedio)
print("\tpathCsvReducido = %s" % pathCsvReducido)
print("\tpathModeloOutliers = %s" % pathModeloOutliers)
print("\tpath_modelo_normalizador = %s" % path_modelo_normalizador)
print("\tpath_indices_out_capa5 = %s" % path_indices_out_capa5)
print("\tpath_modelo_reductor_features = %s" % path_modelo_reductor_features)
print("\tBALANCEO - umbralNecesarioCompensarDesbalanceo = " + str(umbralNecesarioCompensarDesbalanceo))
print("\tBALANCEO - balancearUsandoDownsampling = " + str(balancearUsandoDownsampling))
print("\tBALANCEO - balancearConSmoteSoloTrain = " + str(balancearConSmoteSoloTrain))

#######################################################################################

with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore", message="Bins whose width are too small.*")  # Ignorar los warnings del tramificador (KBinsDiscretizer)

np.random.seed(12345)

print(" **** CAPAS 5 y 6 - Crear almacenar y evaluar varios modelos (para cada subgrupo) ****")
print("Tipo de problema: CLASIFICACION BINOMIAL (target es boolean)")

print("PARAMETROS: ")
pathFeaturesSeleccionadas = dir_subgrupo + "FEATURES_SELECCIONADAS.csv"
umbralCasosSuficientesClasePositiva = 50
umbralProbTargetTrue = 0.50  # IMPORTANTE: umbral para decidir si el target es true o false
granProbTargetUno = 100  # De todos los target=1, nos quedaremos con los granProbTargetUno (en tanto por cien) MAS probables. Un valor de 100 o mayor anula este parámetro
umbralFeaturesCorrelacionadas = varianza  # Umbral aplicado para descartar features cuya correlacion sea mayor que él
cv_todos = 25  # CROSS_VALIDATION: número de iteraciones. Sirve para evitar el overfitting
fraccion_train = 0.50  # Fracción de datos usada para entrenar
fraccion_test = 0.25  # Fracción de datos usada para testear (no es validación)
fraccion_valid = 1.00 - (fraccion_train + fraccion_test)

print("\tumbralCasosSuficientesClasePositiva (positivos en pasado de train+test+valid) = " + str(
    umbralCasosSuficientesClasePositiva))
print("\tumbralProbTargetTrue = " + str(umbralProbTargetTrue))
print("\tgranProbTargetUno = " + str(granProbTargetUno))

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
pathCsvCompletoConFeaturesPython = dir_subgrupo + "COMPLETO_ConFeaturesPython.csv"
pathColumnasConDemasiadosNulos = (dir_subgrupo + "columnasDemasiadosNulos.txt").replace("futuro", "pasado")  # lo guardamos siempre en el pasado

print("\tdir_subgrupo: %s" % dir_subgrupo)
print("\tmodoTiempo: %s" % modoTiempo)
print("\tdesplazamientoAntiguedad: %s" % desplazamientoAntiguedad)
print("\tpathCsvReducido: %s" % pathCsvReducido)
print("\tdir_subgrupo_img = %s" % dir_subgrupo_img)
print("\tumbralFeaturesCorrelacionadas = " + str(umbralFeaturesCorrelacionadas))

################## MAIN ###########################################################
if __name__ != '__main__':
    print("Solo queremos ejecutar este script en modo MAIN, pero se está ejecutando desde fuera como librería, con __name__: " + __name__ + "    Saliendo...")
    exit(-1)

pathCsvEntradaEsCorrecto = (pathCsvCompleto.endswith('.csv') and os.path.isfile(pathCsvCompleto) and os.stat(pathCsvCompleto).st_size > 0)
if pathCsvEntradaEsCorrecto == False:
    print("El path del CSV de entrada NO es correcto: " + pathCsvCompleto + "   Saliendo...")
    exit(-1)

check_file_and_exit(pathCsvCompleto)

print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " ----- leerFeaturesyTarget ------")
print("PARAMS --> " + pathCsvCompleto + "|" + dir_subgrupo_img + "|" + str(compatibleParaMuchasEmpresas) + "|" + pathModeloOutliers + "|" + modoTiempo + "|" + str(modoDebug)
      + "|" + str(maxFilasEntrada))

print("Cargar datos (CSV): "+pathCsvCompleto)
entradaFeaturesYTarget = pd.read_csv(filepath_or_buffer=pathCsvCompleto, sep='|',
                                     error_bad_lines=False,warn_bad_lines=False)
print("entradaFeaturesYTarget (LEIDO): " + str(entradaFeaturesYTarget.shape[0]) + " x " + str(entradaFeaturesYTarget.shape[1]))
C5C6ManualFunciones.mostrarEmpresaConcreta(entradaFeaturesYTarget, DEBUG_EMPRESA, DEBUG_MES, DEBUG_DIA, 30)

### IMPORTANTE: entrenar quitando velas muy antiguas: las de hace mas de 4 meses (90 velas)
entradaFeaturesYTarget = entradaFeaturesYTarget[entradaFeaturesYTarget['antiguedad'] < UMBRAL_VELASMUYANTIGUASELIMINABLES]

################### SE CREAN VARIABLES ADICIONALES #########################

# Se mete la variable dia_aux, mes_aux, low_aux, volume_aux, high_aux,
# ya que las originales se borran más adelante, pero son útiles
entradaFeaturesYTarget['dia_aux'] = entradaFeaturesYTarget['dia']
entradaFeaturesYTarget['mes_aux'] = entradaFeaturesYTarget['mes']
entradaFeaturesYTarget['low_aux'] = entradaFeaturesYTarget['low']
entradaFeaturesYTarget['volumen_aux'] = entradaFeaturesYTarget['volumen']
entradaFeaturesYTarget['high_aux'] = entradaFeaturesYTarget['high']

# Se crea un RSI-14 en python
entradaFeaturesYTarget['RSI-python'] = C5C6ManualFunciones.relative_strength_idx(entradaFeaturesYTarget).fillna(0)

# ##################### Variables del SP500 ################
# C5C6ManualFunciones.aniadirColumnasDependientesSP500()

# ##################### Variables de TWITTER ################
# C5C6ManualFunciones.aniadirColumnasDeTwitter()

######################
# Se guarda el conjunto de datos con las nuevas features añadidas en Python. Se usará sólo para depuración
print("Se guarda el conjunto de features, que incluyen las de Python, en: " + pathCsvCompletoConFeaturesPython)
entradaFeaturesYTarget.to_csv(pathCsvCompletoConFeaturesPython, index=False, sep='|', float_format='%.4f')
#################################

############# MUY IMPORTANTE: creamos el IDENTIFICADOR UNICO DE FILA, que será el indice!!
print("MUY IMPORTANTE: creamos el IDENTIFICADOR UNICO DE FILA, que será el indice!!")
nuevoindice = entradaFeaturesYTarget["empresa"].astype(str) + "_" + entradaFeaturesYTarget["anio"].astype(str) \
              + "_" + entradaFeaturesYTarget["mes"].astype(str) + "_" + entradaFeaturesYTarget["dia"].astype(str)
entradaFeaturesYTarget = entradaFeaturesYTarget.reset_index().set_index(nuevoindice, drop=True)
entradaFeaturesYTarget = entradaFeaturesYTarget.drop('index', axis=1)  # columna index no util ya

C5C6ManualFunciones.mostrarEmpresaConcreta(entradaFeaturesYTarget, DEBUG_EMPRESA, DEBUG_MES, DEBUG_DIA, 30)
###########################################################################

entradaFeaturesYTarget.sort_index(
    inplace=True)  # Reordenando segun el indice (para facilitar el testIntegracion)
entradaFeaturesYTarget.to_csv(pathCsvIntermedio + ".entrada", index=True, sep='|', float_format='%.4f')  # NO BORRAR: UTIL para testIntegracion

if int(maxFilasEntrada) < entradaFeaturesYTarget.shape[0]:
    print("entradaFeaturesYTarget (APLICANDO MAXIMO): " + str(entradaFeaturesYTarget.shape[0]) + " x " + str(entradaFeaturesYTarget.shape[1]))
    entradaFeaturesYTarget = entradaFeaturesYTarget.sample(int(maxFilasEntrada), replace=False)

# Reordenando segun el indice (para facilitar el testIntegracion)
entradaFeaturesYTarget.sort_index(inplace=True)
entradaFeaturesYTarget.to_csv(pathCsvIntermedio + ".entrada_tras_maximo.csv", index=True, sep='|')  # NO BORRAR: UTIL para testIntegracion
entradaFeaturesYTarget.to_csv(pathCsvIntermedio + ".entrada_tras_maximo_INDICES.csv", columns=[])  # NO BORRAR: UTIL para testIntegracion

num_nulos_por_fila_1 = np.logical_not(entradaFeaturesYTarget.isnull()).sum()
indiceFilasFuturasTransformadas1 = entradaFeaturesYTarget.index.values  # DEFAULT
indiceFilasFuturasTransformadas2 = entradaFeaturesYTarget.index.values  # DEFAULT

################# Borrado de columnas nulas enteras ##########
print("MISSING VALUES (COLUMNAS) - Borramos las columnas (features) que sean siempre NaN...")
entradaFeaturesYTarget2 = entradaFeaturesYTarget.dropna(axis=1, how='all')  # Borrar COLUMNA si TODOS sus valores tienen NaN
print("entradaFeaturesYTarget2 (columnas nulas borradas): " + str(entradaFeaturesYTarget2.shape[0]) + " x " + str(entradaFeaturesYTarget2.shape[1]))
# entradaFeaturesYTarget2.to_csv(path_csv_completo + "_TEMP01", index=True, sep='|')  # UTIL para testIntegracion

################# Borrado de filas que tengan algun hueco (tratamiento espacial para el futuro con sus columnas: TARGET y otras) #####
print("MISSING VALUES (FILAS)...")

if modoTiempo == "futuro":
    print("Nos quedamos solo con las velas con antiguedad=0 (futuras)...")
    entradaFeaturesYTarget2 = entradaFeaturesYTarget2[entradaFeaturesYTarget2.antiguedad == 0]
    print("entradaFeaturesYTarget2: " + str(entradaFeaturesYTarget2.shape[0]) + " x " + str(entradaFeaturesYTarget2.shape[1]))
    # print(entradaFeaturesYTarget2.head())

print("Porcentaje de MISSING VALUES en cada columna del dataframe de entrada (mostramos las que superen " + str(
    100 * UMBRAL_COLUMNAS_DEMASIADOS_NULOS) + "%):")
missing = pd.DataFrame(entradaFeaturesYTarget2.isnull().sum()).rename(columns={0: 'total'})
missing['percent'] = missing['total'] / len(entradaFeaturesYTarget2)  # Create a percentage missing
missing_df = missing.sort_values('percent', ascending=False)
missing_df = missing_df[missing_df['percent'] > UMBRAL_COLUMNAS_DEMASIADOS_NULOS]

if 'TARGET' in missing_df.index:  # Por si acaso TARGET es una de las columnas, la excluimos
    missing_df = missing_df.drop('TARGET')

# print(tabulate(missing_df.head(), headers='keys', tablefmt='psql'))  # .drop('TARGET')

# print("Pasado o Futuro: Transformacion en la que borro filas. Por tanto, guardo el indice...")
indiceFilasFuturasTransformadas1 = entradaFeaturesYTarget2.index.values

print(
    "Borrar columnas especiales (idenficadores de fila): empresa | antiguedad | mercado | anio | mes | dia | hora | minuto...")
entradaFeaturesYTarget2 = entradaFeaturesYTarget2.drop('empresa', axis=1).drop('antiguedad', axis=1) \
    .drop('mercado', axis=1).drop('anio', axis=1).drop('mes', axis=1).drop('dia', axis=1) \
    .drop('hora', axis=1).drop('minuto', axis=1) \
    .drop('dia_aux', axis=1).drop('mes_aux', axis=1).drop('low_aux', axis=1).drop('volumen_aux', axis=1) \
    .drop('high_aux', axis=1)

print("Borrar columnas dinamicas que no aportan nada: volumen | high | low | close | open ...")
entradaFeaturesYTarget2 = entradaFeaturesYTarget2.drop('volumen', axis=1).drop('high', axis=1).drop('low', axis=1).drop('close', axis=1).drop('open', axis=1)

print("entradaFeaturesYTarget2 (tras quitar columnas dinamicas identificadoras y con valores absolutos): " + str(entradaFeaturesYTarget2.shape[0]) + " x " + str(entradaFeaturesYTarget2.shape[1]))
C5C6ManualFunciones.mostrarEmpresaConcretaConFilter(entradaFeaturesYTarget2, DEBUG_FILTRO,
                                                    "entradaFeaturesYTarget2 (Tras quitar columnas dinamicas identificadoras y con valores absolutos)")

if modoTiempo == "pasado":
    print("Borrar COLUMNAS dinamicas con demasiados nulos (umbral = " + str(UMBRAL_COLUMNAS_DEMASIADOS_NULOS) + ")...")
    columnasDemasiadosNulos = missing_df.index.values
    print("columnasDemasiadosNulos: " + columnasDemasiadosNulos)

    print("Guardando columnasDemasiadosNulos en: " + pathColumnasConDemasiadosNulos)
    pickle.dump(columnasDemasiadosNulos, open(pathColumnasConDemasiadosNulos, 'wb'))

    # print(tabulate(entradaFeaturesYTarget2.head(), headers='keys', tablefmt='psql'))
    entradaFeaturesYTarget2 = entradaFeaturesYTarget2.drop(columnasDemasiadosNulos, axis=1)
    print("entradaFeaturesYTarget2: " + str(entradaFeaturesYTarget2.shape[0]) + " x " + str(entradaFeaturesYTarget2.shape[1]))
    C5C6ManualFunciones.mostrarEmpresaConcretaConFilter(entradaFeaturesYTarget2, DEBUG_FILTRO, "entradaFeaturesYTarget2 (tras quitar las columnas con demasiados nulos)")

    print(
        "MISSING VALUES (FILAS) - Borramos las FILAS que tengan 1 o mas valores NaN porque son huecos que no deberían estar...")
    entradaFeaturesYTarget3 = entradaFeaturesYTarget2.dropna(axis=0, how='any')  # Borrar FILA si ALGUNO sus valores tienen NaN
    print("entradaFeaturesYTarget3: " + str(entradaFeaturesYTarget3.shape[0]) + " x " + str(entradaFeaturesYTarget3.shape[1]))
    C5C6ManualFunciones.mostrarEmpresaConcretaConFilter(entradaFeaturesYTarget3, DEBUG_FILTRO, "entradaFeaturesYTarget3")

elif modoTiempo == "futuro":
    print(
        "Borrar COLUMNAS dinamicas con demasiados nulos desde un fichero del pasado..." + pathColumnasConDemasiadosNulos)
    columnasDemasiadosNulos = pickle.load(open(pathColumnasConDemasiadosNulos, 'rb'))
    # print("columnasDemasiadosNulos: "); print(columnasDemasiadosNulos)
    # print(tabulate(entradaFeaturesYTarget2.head(), headers='keys', tablefmt='psql'))
    columns_to_drop = [col for col in columnasDemasiadosNulos if col in entradaFeaturesYTarget2.columns]
    entradaFeaturesYTarget2 = entradaFeaturesYTarget2.drop(columns_to_drop, axis=1)
    print("entradaFeaturesYTarget2: " + str(entradaFeaturesYTarget2.shape[0]) + " x " + str(entradaFeaturesYTarget2.shape[1]))

    print("MISSING VALUES (FILAS) - Para el FUTURO; el target es NULO siempre...")
    if "TARGET" in entradaFeaturesYTarget2.columns:
        entradaFeaturesYTarget2 = entradaFeaturesYTarget2.drop('TARGET', axis=1)

    # print(tabulate(entradaFeaturesYTarget2.head(), headers='keys', tablefmt='psql'))
    print("MISSING VALUES (FILAS) - Para el FUTURO, borramos las filas que tengan ademas otros NULOS...")
    entradaFeaturesYTarget3 = entradaFeaturesYTarget2
    print("entradaFeaturesYTarget3: " + str(entradaFeaturesYTarget3.shape[0]) + " x " + str(entradaFeaturesYTarget3.shape[1]))
    print("Por si acaso en el futuro viniera una columna con todo NaN, ponemos un 0 a todo...")
    entradaFeaturesYTarget3 = entradaFeaturesYTarget3.replace(np.nan, 0)

print("Pasado o futuro: Transformacion en la que he borrado filas. Por tanto, guardo el indice...")
indiceFilasFuturasTransformadas2 = entradaFeaturesYTarget3.index.values
print("indiceFilasFuturasTransformadas2: " + str(indiceFilasFuturasTransformadas2.shape[0]))
# print(indiceFilasFuturasTransformadas2)

print("entradaFeaturesYTarget3 (filas con algun nulo borradas):" + str(entradaFeaturesYTarget3.shape[0]) + " x " + str(entradaFeaturesYTarget3.shape[1]))

entradaFeaturesYTarget3.to_csv(pathCsvIntermedio + ".sololascompletas.csv", index=True, sep='|')  # NO BORRAR: UTIL para testIntegracion
entradaFeaturesYTarget3.to_csv(pathCsvIntermedio + ".sololascompletas_INDICES.csv", columns=[])  # NO BORRAR: UTIL para testIntegracion

print("Limpiar OUTLIERS...")
# URL: https://scikit-learn.org/stable/modules/outlier_detection.html
# Pasado y futuro:
print("Comprobando que el dataframe de entrada tenga datos...")
num_filas_antesdeaouliers = entradaFeaturesYTarget3.shape[0]
if (num_filas_antesdeaouliers == 0):
    raise RuntimeError("Hay 0 filas en el dataframe de entrada: entradaFeaturesYTarget3 Salimos...")

if modoTiempo == "pasado":
    detector_outliers = IsolationForest()
    df3aux = entradaFeaturesYTarget3.drop('TARGET', axis=1, errors='ignore')
    detector_outliers.fit(df3aux)  # fit 10 trees
    pickle.dump(detector_outliers, open(pathModeloOutliers, 'wb'))
else:
    df3aux = entradaFeaturesYTarget3  # FUTURO

# Pasado y futuro:
print("Comprobando que el dataframe de entrada tenga datos...")
num_filas = df3aux.shape[0]
if (num_filas == 0):
    raise RuntimeError("Hay 0 filas en el dataframe de entrada: df3aux Salimos...")

print("Cargando modelo detector de outliers: " + pathModeloOutliers)
detector_outliers = pickle.load(open(pathModeloOutliers, 'rb'))
flagAnomaliasDf = pd.DataFrame({'marca_anomalia': detector_outliers.predict(df3aux)})  # vale -1 es un outlier; si es un 1, no lo es

if limpiarOutliers is True:
    indice3 = entradaFeaturesYTarget3.index  # lo guardo para pegarlo luego
    entradaFeaturesYTarget3.reset_index(drop=True, inplace=True)
    flagAnomaliasDf.reset_index(drop=True, inplace=True)
    entradaFeaturesYTarget4 = pd.concat([entradaFeaturesYTarget3, flagAnomaliasDf], axis=1)  # Column Bind, manteniendo el índice del DF izquierdo
    entradaFeaturesYTarget4.set_index(indice3, inplace=True)  # ponemos el indice que tenia el DF de la izquierda

    entradaFeaturesYTarget4 = entradaFeaturesYTarget4.loc[entradaFeaturesYTarget4['marca_anomalia'] == 1]  # Cogemos solo las que no son anomalias
    entradaFeaturesYTarget4 = entradaFeaturesYTarget4.drop('marca_anomalia', axis=1)  # Quitamos la columna auxiliar

else:
    entradaFeaturesYTarget4 = entradaFeaturesYTarget3

print("entradaFeaturesYTarget4 (sin outliers):" + str(entradaFeaturesYTarget4.shape[0]) + " x " + str(entradaFeaturesYTarget4.shape[1]))
C5C6ManualFunciones.mostrarEmpresaConcretaConFilter(entradaFeaturesYTarget4, DEBUG_FILTRO, "entradaFeaturesYTarget4")
entradaFeaturesYTarget4.to_csv(pathCsvIntermedio + ".sinoutliers.csv", index=True, sep='|')  # NO BORRAR: UTIL para testIntegracion
entradaFeaturesYTarget4.to_csv(pathCsvIntermedio + ".sinoutliers_INDICES.csv", columns=[])  # NO BORRAR: UTIL para testIntegracion

# ENTRADA: features (+ target)
if modoTiempo == "pasado":
    featuresFichero = entradaFeaturesYTarget4.drop('TARGET', axis=1, errors='ignore')  # default
    targetsFichero = (entradaFeaturesYTarget4[['TARGET']] == 1)  # default
else:
    featuresFichero = entradaFeaturesYTarget4

if compatibleParaMuchasEmpresas is False or modoTiempo == "futuro":
    # Si hay POCAS empresas
    print("POCAS EMPRESAS (modoTiempo=" + modoTiempo + ")...")
    if modoTiempo == "pasado":
        featuresFichero = entradaFeaturesYTarget4.drop('TARGET', axis=1)
        targetsFichero = (entradaFeaturesYTarget4[['TARGET']] == 1)  # Convierto de int a boolean
    elif modoTiempo == "futuro":
        featuresFichero = entradaFeaturesYTarget4
        targetsFichero = pd.DataFrame({'TARGET': []})  # DEFAULT: Vacio (caso futuro)
else:
    # SOLO PARA EL PASADO Si hay MUCHAS empresas (UNDER-SAMPLING para reducir los datos -útil para miles de empresas, pero puede quedar sobreentrenado, si borro casi todas las minoritarias-)
    print("MUCHAS EMPRESAS (modoTiempo=" + modoTiempo + ")...")
    print("NO balanceamos clases en capa 5 (pero seguramente sí en capa 6 solo sobre dataset de TRAIN)!!!")
    ift_minoritaria = entradaFeaturesYTarget4[(entradaFeaturesYTarget4.TARGET == True)]
    ift_mayoritaria = entradaFeaturesYTarget4[(entradaFeaturesYTarget4.TARGET == False)]
    print("Tasa de desbalanceo entre clases = " + str(ift_mayoritaria.shape[0]) + "/" + str(ift_minoritaria.shape[0]) + " = " + str(ift_mayoritaria.shape[0] / ift_minoritaria.shape[0]))
    entradaFeaturesYTarget5 = entradaFeaturesYTarget4
    C5C6ManualFunciones.mostrarEmpresaConcretaConFilter(entradaFeaturesYTarget5, DEBUG_FILTRO, "entradaFeaturesYTarget5")

    # entradaFeaturesYTarget5.to_csv(path_csv_completo + "_TEMP05", index=True, sep='|')  # UTIL para testIntegracion
    featuresFichero = entradaFeaturesYTarget5.drop('TARGET', axis=1)
    targetsFichero = entradaFeaturesYTarget5[['TARGET']]
    targetsFichero = (targetsFichero[['TARGET']] == 1)  # Convierto de int a boolean

    print("entradaFeaturesYTarget5 (sin outliers):" + str(entradaFeaturesYTarget5.shape[0]) + " x " + str(entradaFeaturesYTarget5.shape[1]))

##################################################
if modoDebug and modoTiempo == "pasado":
    C5C6ManualFunciones.pintarFuncionesDeDensidad(featuresFichero, dir_subgrupo_img, dibujoBins, "sin nulos, pero antes de normalizar")

################################### NORMALIZACIÓN ####################################################
if evitarNormalizarNiTramificar is False:
    # NORMALIZAR, PERO SIN TRAMIFICAR: leer apartado 4.3 de https://eprints.ucm.es/56355/1/TFM_MPP_Jul19%20%281%29Palau.pdf
    featuresFichero3 = C5C6ManualML.normalizar(path_modelo_normalizador, featuresFichero, modoTiempo, pathCsvIntermedio, modoDebug, dir_subgrupo_img, dibujoBins, DEBUG_FILTRO)
else:
    print("NO NORMALIZAR y NO TRAMIFICAR")
    featuresFichero3 = featuresFichero

print("Tras la normalización (opcional) queda --> featuresFichero3: " + str(featuresFichero3.shape[0]) + " x " + str(featuresFichero3.shape[1]) + "  y  " + "targetsFichero: " +
      str(targetsFichero.shape[0]) + " x " + str(targetsFichero.shape[1]))

##############################################################################
C5C6ManualML.comprobarSuficientesClasesDelTarget(targetsFichero, modoTiempo)

##############################################################################
print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " ----- reducirFeaturesYGuardar ------")
print("\tpath_modelo_reductor_features --> " + path_modelo_reductor_features)
print("\tpath_modelo_pca --> " + path_modelo_pca)
print("\tpath_pesos_pca --> " + path_pesos_pca)
print("\tfeaturesFichero3: " + str(featuresFichero3.shape[0]) + " x " + str(featuresFichero3.shape[1]))
print("\ttargetsFichero: " + str(targetsFichero.shape[0]) + " x " + str(targetsFichero.shape[1]))
print("\tpathCsvReducido --> " + pathCsvReducido)
print("\tpathCsvFeaturesElegidas --> " + pathCsvFeaturesElegidas)
print("\tvarianza (PCA) --> " + str(varianza))
print("\tdir_subgrupo_img --> " + dir_subgrupo_img)
print("\tmodoTiempo: " + modoTiempo)
print("\tmaxFeatReducidas: " + maxFeatReducidas)

######################## SIN RFECV ###############################
featuresFichero3Elegidas = featuresFichero3
columnasSeleccionadas = featuresFichero3.columns
####################### FIN SIN RFECV ###############################

print(
    "Features seleccionadas (tras el paso de RFECV, cuya aplicacion es opcional) escritas en: " + pathCsvFeaturesElegidas)
featuresFichero3Elegidas.head(1).to_csv(pathCsvFeaturesElegidas, index=False, sep='|', float_format='%.4f')

########### PCA: base de funciones ortogonales (con combinaciones de features) ########
print("** PCA (Principal Components Algorithm) **")

if modoTiempo == "pasado":
    # print("Usando PCA, creamos una NUEVA BASE DE FEATURES ORTOGONALES y cogemos las que tengan un impacto agregado sobre el "+str(varianza)+"% de la varianza del target. Descartamos el resto.")
    # modelo_pca_subgrupo = PCA(n_components=varianza, svd_solver='full')  # Varianza acumulada sobre el target

    modelo_pca_subgrupo = PCA(n_components='mle', svd_solver='full')  # Metodo "MLE de Minka": https://vismod.media.mit.edu/tech-reports/TR-514.pdf

    # modelo_pca_subgrupo = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0,
    #                            n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07,
    #                            metric='euclidean', init='random', verbose=0, random_state=None,
    #                            method='barnes_hut', angle=0.5,
    #                            n_jobs=-1)  # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    print(modelo_pca_subgrupo)
    featuresFichero3_pca = modelo_pca_subgrupo.fit_transform(featuresFichero3Elegidas)
    print("modelo_pca_subgrupo -> Guardando en: " + path_modelo_pca)
    pickle.dump(modelo_pca_subgrupo, open(path_modelo_pca, 'wb'))

else:
    print("modelo_pca_subgrupo -> Leyendo desde: " + path_modelo_pca)
    modelo_pca_subgrupo = pickle.load(open(path_modelo_pca, 'rb'))
    print(modelo_pca_subgrupo)
    featuresFichero3_pca = modelo_pca_subgrupo.transform(featuresFichero3Elegidas)

print("Dimensiones del dataframe tras PCA: " + str(featuresFichero3_pca.shape[0]) + " x " + str(featuresFichero3_pca.shape[1]))

# print("Las features están ya normalizadas, reducidas y en base ortogonal PCA. DESCRIBIMOS lo que hemos hecho y GUARDAMOS el dataset.")
num_columnas_pca = featuresFichero3_pca.shape[1]
columnas_pca = ["pca_" + f"{i:0>2}" for i in range(num_columnas_pca)]  # Hacemos left padding con la funcion f-strings
featuresFichero3_pca_df = pd.DataFrame(featuresFichero3_pca, columns=columnas_pca, index=featuresFichero3.index)
# print(tabulate(featuresFichero3_pca_df.head(), headers='keys', tablefmt='psql'))  # .drop('TARGET')
featuresFichero3Elegidas = featuresFichero3_pca_df

print("Matriz de pesos de las features en la base de funciones PCA: " + path_pesos_pca)
pcaMatriz = pd.DataFrame(modelo_pca_subgrupo.components_)
pcaMatriz.columns = columnasSeleccionadas
columnas_pca_df = pd.DataFrame(columnas_pca)
pcaMatriz = pd.concat([columnas_pca_df, pcaMatriz], axis=1)
pcaMatriz.to_csv(path_pesos_pca, index=False, sep='|', float_format='%.4f')

### Guardar a fichero
# print("Muestro las features + targets antes de juntarlas...")
# print("FEATURES (sample):")
# print(featuresFichero3Elegidas.head())
print("featuresFichero3Elegidas: " + str(featuresFichero3Elegidas.shape[0]) + " x " + str(featuresFichero3Elegidas.shape[1]))
# print("TARGETS (sample):")
# print(targetsFichero.head())

featuresytargets = pd.concat([featuresFichero3Elegidas.reset_index(drop=True), targetsFichero.reset_index(drop=True)], axis=1)  # Column bind
featuresytargets.set_index(featuresFichero3Elegidas.index, inplace=True)
# print("FEATURES+TARGETS juntas (sample):")
# print(featuresytargets.head())
print("Justo antes de guardar, featuresytargets: " + str(featuresytargets.shape[0]) + " x " + str(featuresytargets.shape[1]))
C5C6ManualFunciones.mostrarEmpresaConcretaConFilter(featuresytargets, DEBUG_FILTRO, "featuresytargets tras PCA")
featuresytargets.to_csv(pathCsvReducido, index=True, sep='|', float_format='%.4f')

print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " ------------------------------- FIN de capa 5 --------------------------------")
####################################################################

# GANADOR DEL SUBGRUPO (acumuladores)
ganador_nombreModelo = "NINGUNO"
ganador_metrica = 0
ganador_metrica_avg = 0
ganador_grid_mejores_parametros = []
pathListaColumnasCorreladasDrop = (dir_subgrupo + "columnas_correladas_drop" + ".txt").replace("futuro", "pasado")  # lo guardamos siempre en el pasado

if (modoTiempo == "pasado" and pathCsvReducido.endswith('.csv') and os.path.isfile(pathCsvReducido) and os.stat(pathCsvReducido).st_size > 0):

    print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " Capa 6 - Modo PASADO")
    inputFeaturesyTarget = pd.read_csv(pathCsvReducido, index_col=0, sep='|')  # La columna 0 contiene el indice
    print("inputFeaturesyTarget: " + str(inputFeaturesyTarget.shape[0]) + " x " + str(inputFeaturesyTarget.shape[1]))
    C5C6ManualFunciones.mostrarEmpresaConcretaConFilter(inputFeaturesyTarget, DEBUG_FILTRO, "inputFeaturesyTarget entrada capa 6")

    print("BALANCEAR los casos positivos y negativos, haciendo downsampling de la clase mayoritaria...")
    # Explicaciones: https://elitedatascience.com/imbalanced-classes
    ift_mayoritaria = inputFeaturesyTarget[inputFeaturesyTarget.TARGET == False]  # En este caso los mayoritarios son los False
    ift_minoritaria = inputFeaturesyTarget[inputFeaturesyTarget.TARGET == True]
    print("ift_mayoritaria: " + str(ift_mayoritaria.shape[0]) + " x " + str(ift_mayoritaria.shape[1]))
    print("ift_minoritaria: " + str(ift_minoritaria.shape[0]) + " x " + str(ift_minoritaria.shape[1]))
    tasaDesbalanceoAntes = round(ift_mayoritaria.shape[0] / ift_minoritaria.shape[0], 2)
    print("Tasa de desbalanceo entre clases (antes de balancear INICIO) = " + str(ift_mayoritaria.shape[0]) + "/" + str(ift_minoritaria.shape[0]) + " = " + str(tasaDesbalanceoAntes))
    num_muestras_minoria = ift_minoritaria.shape[0]

    casosInsuficientes = (num_muestras_minoria < umbralCasosSuficientesClasePositiva)
    if (casosInsuficientes):
        print("Numero de casos en clase minoritaria es INSUFICIENTE: " + str(num_muestras_minoria) + " (umbral=" + str(
            umbralCasosSuficientesClasePositiva) + "). Así que abandonamos este dataset y seguimos")

    else:
        print("Clase MINORITARIA tiene suficientes casos. En el dataframe se coloca primero todas las mayoritarias y despues todas las minoritarias")
        ift_juntas = pd.concat([ift_mayoritaria.reset_index(drop=True), ift_minoritaria.reset_index(drop=True)], axis=0)  # Row bind
        indices_juntos = ift_mayoritaria.index.append(ift_minoritaria.index)  # Row bind
        ift_juntas.set_index(indices_juntos, inplace=True)
        print("Las clases juntas son:")
        print("ift_juntas: " + str(ift_juntas.shape[0]) + " x " + str(ift_juntas.shape[1]))
        C5C6ManualFunciones.mostrarEmpresaConcretaConFilter(ift_juntas, DEBUG_FILTRO, "ift_juntas")
        ift_juntas.to_csv(pathCsvIntermedio + ".trasbalancearclases.csv", index=True, sep='|', float_format='%.4f')  # NO BORRAR: UTIL para testIntegracion
        ift_juntas.to_csv(pathCsvIntermedio + ".trasbalancearclases_INDICES.csv", columns=[])  # NO BORRAR: UTIL para testIntegracion
        C5C6ManualFunciones.describirConPandasProfiling(modoDebug, ift_juntas, dir_subgrupo)

        ###################### Matriz de correlaciones y quitar features correladas ###################
        C5C6ManualML.matrizCorrelacionesYquitarFaturesCorreladas(ift_juntas, umbralFeaturesCorrelacionadas, pathListaColumnasCorreladasDrop, pathFeaturesSeleccionadas)

        ############################## DIVISIÓN DE DATOS: TRAIN, TEST, VALIDACIÓN ##########################
        ds_train, ds_test, ds_validacion, ds_train_f, ds_train_t, ds_test_f, ds_test_t, ds_validac_f, ds_validac_t, ds_train_f_sinsmote, ds_train_t_sinsmote = C5C6ManualFunciones.splitTrainTestValidation(
            modoTiempo, ift_juntas, fraccion_train, fraccion_test, fraccion_valid, balancearConSmoteSoloTrain, umbralNecesarioCompensarDesbalanceo, balancearUsandoDownsampling, DEBUG_FILTRO)

        ############## ENTRENAMIENTO #####################
        pathModelo, nombreModelo = C5C6ManualML.entrenarModeloModoPasado(dir_subgrupo, ds_train_f, ds_train_t, ds_test_f, ds_test_t)

        ############################## APERTURA DE MODELO YA ENTRENADO #########################
        modeloPredictivoEntrenado = pickle.load(open(pathModelo, 'rb'))

        ############################## PREDICCION y CALCULO DE LAS METRICAS EN TEST Y VALID ####################
        train_t_predicho, test_t_predicho, validac_t_predicho, ganador_metrica, ganador_metrica_avg, ganador_nombreModelo, ganador_grid_mejores_parametros = C5C6ManualML.calcularMetricasModeloEntrenado(
            id_subgrupo, modeloPredictivoEntrenado, ds_train_f_sinsmote, ds_train_t_sinsmote, ds_train_t, ds_test_f, ds_test_t, ds_validac_f, ds_validac_t, dir_subgrupo_img, pathCsvIntermedio,
            tasaDesbalanceoAntes, nombreModelo, ganador_metrica)

        ######################################################################################################################

        print("********* GANADOR de subgrupo *************")
        num_positivos_train = ds_train_t_sinsmote[(ds_train_t_sinsmote == True)].count().values[0]
        num_positivos_test = ds_test_t[(ds_test_t == True)].count().values[0]
        num_positivos_validac = ds_validac_t[(ds_validac_t == True)].count().values[0]
        print("\tnum_positivos_train="+str(num_positivos_train))
        print("\tnum_positivos_test=" + str(num_positivos_test))
        print("\tnum_positivos_validac=" + str(num_positivos_validac))

        print("PASADO -> " + id_subgrupo + " (num features = " + str(ds_train_f.shape[1]) + ")" + " -> Modelo ganador = " + ganador_nombreModelo + " --> METRICA = " + str(
            round(ganador_metrica, 4)) + " (avg_precision = " + str(round(ganador_metrica_avg, 4)) + ")")

        print("PASADO -> " + id_subgrupo + " TASA DE MEJORA DE PRECISION RESPECTO A RANDOM: ", round(ganador_metrica / (1 / (1 + tasaDesbalanceoAntes)), 2))

        print("Hiperparametros:")
        print(ganador_grid_mejores_parametros)
        pathModeloGanadorDeSubgrupoOrigen = dir_subgrupo + ganador_nombreModelo + ".modelo"
        pathModeloGanadorDeSubgrupoDestino = pathModeloGanadorDeSubgrupoOrigen + "_ganador"
        if os.path.exists(pathModeloGanadorDeSubgrupoOrigen):

            copyfile(pathModeloGanadorDeSubgrupoOrigen, pathModeloGanadorDeSubgrupoDestino)
            print("Modelo ganador guardado en: " + pathModeloGanadorDeSubgrupoDestino)

            ############### COMPROBACIÓN MANUAL DE LA PRECISIÓN ######################################
            # Cruce por indice para sacar los nombres de las empresas de cada fila
            df_train_empresas_index = pd.merge(entradaFeaturesYTarget, ds_train, left_index=True, right_index=True)
            df_test_empresas_index = pd.merge(entradaFeaturesYTarget, ds_test, left_index=True, right_index=True)
            df_valid_empresas_index = pd.merge(entradaFeaturesYTarget, ds_validacion, left_index=True, right_index=True)

            ds_train_f_temp = ds_train.drop('TARGET', axis=1)
            df_train_f_sinsmote = pd.DataFrame(ds_train_f_sinsmote, columns=ds_train_f_temp.columns, index=ds_train_f_temp.index)  # Indice que tenia antes de SMOTE

            C5C6ManualFunciones.comprobarPrecisionManualmente(ds_train_t_sinsmote, train_t_predicho, "TRAIN (forzado)", id_subgrupo, df_train_f_sinsmote, dir_subgrupo,
                                                              DEBUG_FILTRO)  # ds_train_t tiene SMOTE!!!
            C5C6ManualFunciones.comprobarPrecisionManualmente(ds_test_t, test_t_predicho, "TEST", id_subgrupo, df_test_empresas_index, dir_subgrupo, DEBUG_FILTRO)
            C5C6ManualFunciones.comprobarPrecisionManualmente(ds_validac_t, validac_t_predicho, "VALIDACION", id_subgrupo, df_valid_empresas_index, dir_subgrupo, DEBUG_FILTRO)
            ######################################################################################################################
        else:
            print("No se ha guardado modelo:" + pathModeloGanadorDeSubgrupoOrigen)

elif (modoTiempo == "futuro" and pathCsvReducido.endswith('.csv') and os.path.isfile(pathCsvReducido) and os.stat(pathCsvReducido).st_size > 0):

    print("Capa 6 - Modo futuro")
    print("pathCsvReducido: " + pathCsvReducido)
    inputFeaturesyTarget = pd.read_csv(pathCsvReducido, index_col=0, sep='|')
    print("inputFeaturesyTarget: " + str(inputFeaturesyTarget.shape[0]) + " x " + str(inputFeaturesyTarget.shape[1]))

    print("Si las hay, eliminamos las features muy correladas (umbral =" + str(umbralFeaturesCorrelacionadas) + ") aprendido en el PASADO.")
    if os.path.exists(pathListaColumnasCorreladasDrop):
        columnasCorreladas = pickle.load(open(pathListaColumnasCorreladasDrop, 'rb'))
        inputFeaturesyTarget.drop(columnasCorreladas, axis=1, inplace=True)
        print(columnasCorreladas)

    # print("Matriz de correlaciones corregida (FUTURO):")
    matrizCorr = inputFeaturesyTarget.corr()
    # print(matrizCorr)

    # print("La columna TARGET que haya en el CSV de entrada no la queremos (es un NULL o False, por defecto), porque la vamos a PREDECIR...")
    inputFeatures = inputFeaturesyTarget.drop('TARGET', axis=1)
    # print(inputFeatures.head())
    print("inputFeatures: " + str(inputFeatures.shape[0]) + " x " + str(inputFeatures.shape[1]))

    print("MISSING VALUES (FILAS) - Borramos las FILAS que tengan 1 o mas valores NaN porque son huecos que no deberían estar...")
    inputFeatures_sinnulos = inputFeatures.dropna(axis=0, how='any')  # Borrar FILA si ALGUNO sus valores tienen NaN

    dir_modelo_predictor_ganador = dir_subgrupo.replace("futuro", "pasado")  # Siempre cojo el modelo entrenado en el pasado
    path_modelo_predictor_ganador = ""
    for file in os.listdir(dir_modelo_predictor_ganador):
        if file.endswith("ganador"):
            path_modelo_predictor_ganador = os.path.join(dir_modelo_predictor_ganador, file)

    print("Cargar modelo PREDICTOR ganador (de la carpeta del pasado, SI EXISTE): " + path_modelo_predictor_ganador)
    if os.path.isfile(path_modelo_predictor_ganador):

        modelo_predictor_ganador = pickle.load(open(path_modelo_predictor_ganador, 'rb'))

        print("Prediciendo...")
        targets_predichos = modelo_predictor_ganador.predict(inputFeatures_sinnulos.to_numpy())
        num_targets_predichos = len(targets_predichos)
        print("Numero de targets_predichos: " + str(num_targets_predichos) + " con numero de TRUEs = " + str(np.sum(targets_predichos, where=["True"])))
        # print("El array de targets contiene:")
        # print(targets_predichos)

        # probabilities
        probs = pd.DataFrame(data=modelo_predictor_ganador.predict_proba(inputFeatures_sinnulos), index=inputFeatures_sinnulos.index)
        probs.columns = ['proba_false', 'proba_true']
        print("FUTURO - Ejemplos de probabilidades al predecir targets true (orden descendiente): ")
        print(tabulate(probs.sort_values("proba_true", ascending=False).head(n=5), headers='keys', tablefmt='psql'))

        # UMBRAL MENOS PROBABLES CON TARGET=1. Cuando el target es 1, se guarda su probabilidad
        # print("El DF llamado probs contiene las probabilidades de predecir un 0 o un 1:")
        # print(probs)
        print("Sobre los targets predichos, se cogen solo aquellos con probabilidad encima del umbral -> umbralProbTargetTrue=" + str(umbralProbTargetTrue) + " y granProbTargetUno=" + str(
            granProbTargetUno))
        probabilidadesEnTargetUnoPeq = probs.iloc[:, 1]  # Cogemos solo la segunda columna: prob de que sea target=1
        probabilidadesEnTargetUnoPeq2 = probabilidadesEnTargetUnoPeq.apply(lambda x: x if (x >= umbralProbTargetTrue) else np.nan)  # Cogemos solo las filas cuya prob_1 > umbral
        probabilidadesEnTargetUnoPeq3 = probabilidadesEnTargetUnoPeq2[np.isnan(probabilidadesEnTargetUnoPeq2[:]) == False]  # Cogemos todos los no nulos (NAN)
        # print("El DF llamado probabilidadesEnTargetUnoPeq3 contiene las probabilidades de los UNO con prob mayor que umbral ("+str(umbralProbTargetTrue)+"):")
        # print(probabilidadesEnTargetUnoPeq3)

        probabilidadesEnTargetUnoPeq4 = probabilidadesEnTargetUnoPeq3.sort_values(ascending=False)  # descendente
        # print("El DF llamado probabilidadesEnTargetUnoPeq4 contiene los indices y probabilidades, tras aplicar umbral INFERIOR: " + str(umbralProbTargetTrue) + ". Son:")
        # print(probabilidadesEnTargetUnoPeq4)

        numfilasSeleccionadas = int(granProbTargetUno * probabilidadesEnTargetUnoPeq4.shape[0] / 100)  # Como están ordenadas en descendente, cojo estas NUM primeras filas
        print("numfilasSeleccionadas: " + str(numfilasSeleccionadas))
        targets_predichosCorregidos_probs = probabilidadesEnTargetUnoPeq4[0:numfilasSeleccionadas]
        targets_predichosCorregidos = targets_predichosCorregidos_probs.apply(lambda x: 1)
        # print("El DF llamado targets_predichosCorregidos contiene los indices y probabilidades, tras aplicar umbral SUPERIOR: top " + str(granProbTargetUno) + " % de muestras. Son:")
        # print(targets_predichosCorregidos)

        print("Guardando targets PREDICHOS en: " + pathCsvPredichos)
        df_predichos = targets_predichosCorregidos.to_frame()
        df_predichos.columns = ['TARGET_PREDICHO']
        df_predichos.to_csv(pathCsvPredichos, index=False, sep='|', float_format='%.4f')  # Capa 6 - Salida (para el validador, sin indice)

        df_predichos_probs = targets_predichosCorregidos_probs.to_frame()
        df_predichos_probs.columns = ['TARGET_PREDICHO_PROB']
        df_predichos_probs.to_csv(pathCsvPredichos + "_humano", index=True, sep='|')  # Capa 6 - Salida (para el humano)

        ############### RECONSTRUCCION DEL CSV FINAL IMPORTANTE, viendo los ficheros de indices #################
        print(
            "Partiendo de COMPLETO.csv llevamos la cuenta de los indices pasando por REDUCIDO.csv y por TARGETS_PREDICHOS.csv para generar el CSV final...")
        print("pathCsvCompleto: " + pathCsvCompleto)
        df_completo = pd.read_csv(pathCsvCompleto, sep='|')  # Capa 5 - Entrada

        print("df_completo: " + str(df_completo.shape[0]) + " x " + str(df_completo.shape[1]))
        print("df_predichos: " + str(df_predichos.shape[0]) + " x " + str(df_predichos.shape[1]))
        print("df_predichos_probs: " + str(df_predichos_probs.shape[0]) + " x " + str(df_predichos_probs.shape[1]))

        # Predichos con columnas: empresa anio mes dia
        indiceDFPredichos = df_predichos.index.values
        df_predichos.insert(0, column="indiceColumna", value=indiceDFPredichos)

        if df_predichos.shape[0] <= 0:
            print("No se ha predicho NINGUN target=1 sabiendo que cogemos el top " + str(granProbTargetUno) + "% elementos que han tenido probabilidad >= " + str(
                umbralProbTargetTrue) + " Por tanto, sa acaba el proceso.")
        else:
            df_predichos[['empresa', 'anio', 'mes', 'dia']] = df_predichos['indiceColumna'].str.split('_', 4, expand=True)
            df_predichos = df_predichos.drop('indiceColumna', axis=1)
            df_predichos = df_predichos.astype({"anio": int, "mes": int, "dia": int})

            indiceDFPredichos = df_predichos_probs.index.values
            df_predichos_probs.insert(0, column="indiceColumna", value=indiceDFPredichos)
            df_predichos_probs[['empresa', 'anio', 'mes', 'dia']] = df_predichos_probs['indiceColumna'].str.split('_', 4, expand=True)
            df_predichos_probs = df_predichos_probs.drop('indiceColumna', axis=1)
            df_predichos_probs = df_predichos_probs.astype({"anio": int, "mes": int, "dia": int})

            print("Juntar COMPLETO con TARGETS PREDICHOS... ")
            df_juntos_1 = pd.merge(df_completo, df_predichos, on=["empresa", "anio", "mes", "dia"], how='left')
            df_juntos_2 = pd.merge(df_juntos_1, df_predichos_probs, on=["empresa", "anio", "mes", "dia"], how='left')

            df_juntos_2['TARGET_PREDICHO'] = (df_juntos_2['TARGET_PREDICHO'] * 1).astype('Int64')  # Convertir de boolean a int64, manteniendo los nulos

            print("Guardando (pathCsvFinalFuturo): " + pathCsvFinalFuturo)
            df_juntos_2.to_csv(pathCsvFinalFuturo, index=False, sep='|', float_format='%.4f')


    else:
        print(
            "No existe el modelo predictor del pasado que necesitamos (" + path_modelo_predictor_ganador + "). Por tanto, no predecimos.")


else:
    print("Los parametros de entrada son incorrectos o el CSV no existe o esta vacio!!")

############################################################
print(" ------------ FIN de capa 6 ----------------")
