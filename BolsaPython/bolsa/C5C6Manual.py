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
from pandas_profiling import ProfileReport
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

print((datetime.datetime.now()).strftime(
    "%Y%m%d_%H%M%S") + " **** CAPA 5  --> Selección de variables/ Reducción de dimensiones (para cada subgrupo) ****")
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

print("dir_subgrupo = %s" % dir_subgrupo)
print("modoTiempo = %s" % modoTiempo)
print("maxFeatReducidas = %s" % maxFeatReducidas)
print("maxFilasEntrada = %s" % maxFilasEntrada)

##################################################################################################
# DEBUG - EMPRESA vigilada en un dia-mes-año
global DEBUG_EMPRESA; DEBUG_EMPRESA = "ADI"
global DEBUG_ANIO; DEBUG_ANIO = 2022
global DEBUG_MES; DEBUG_MES = 2
global DEBUG_DIA; DEBUG_DIA = 17
global DEBUG_FILTRO; DEBUG_FILTRO = DEBUG_EMPRESA + "_" + str(DEBUG_ANIO) + "_" + str(DEBUG_MES) + "_" + str(DEBUG_DIA)
##################################################################################################


varianza = 0.93  # Variacion acumulada de las features PCA sobre el target
UMBRAL_VELASMUYANTIGUASELIMINABLES = (5 * 4.5) * 6  # todas las velas con mas antiguedad que este umbral no se usan para train ni test ni valid. Recom: 90 (4 meses)
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



print("pathCsvCompleto = %s" % pathCsvCompleto)
print("dir_subgrupo_img = %s" % dir_subgrupo_img)
print("pathCsvIntermedio = %s" % pathCsvIntermedio)
print("pathCsvReducido = %s" % pathCsvReducido)
print("pathModeloOutliers = %s" % pathModeloOutliers)
print("path_modelo_normalizador = %s" % path_modelo_normalizador)
print("path_indices_out_capa5 = %s" % path_indices_out_capa5)
print("path_modelo_reductor_features = %s" % path_modelo_reductor_features)
print("BALANCEO - umbralNecesarioCompensarDesbalanceo = " + str(umbralNecesarioCompensarDesbalanceo))
print("BALANCEO - balancearUsandoDownsampling = " + str(balancearUsandoDownsampling))
print("BALANCEO - balancearConSmoteSoloTrain = " + str(balancearConSmoteSoloTrain))
######################## FUNCIONES #######################################################


with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore",
                            message="Bins whose width are too small.*")  # Ignorar los warnings del tramificador (KBinsDiscretizer)

np.random.seed(12345)

print("\n" + (datetime.datetime.now()).strftime(
    "%Y%m%d_%H%M%S") + " **** CAPAS 5 y 6 - Crear almacenar y evaluar varios modelos (para cada subgrupo) ****")
print("Tipo de problema: CLASIFICACION BINOMIAL (target es boolean)")

print("PARAMETROS: ")
pathFeaturesSeleccionadas = dir_subgrupo + "FEATURES_SELECCIONADAS.csv"
umbralCasosSuficientesClasePositiva = 100
umbralProbTargetTrue = 0.50  # IMPORTANTE: umbral para decidir si el target es true o false
granProbTargetUno = 100  # De todos los target=1, nos quedaremos con los granProbTargetUno (en tanto por cien) MAS probables. Un valor de 100 o mayor anula este parámetro
umbralFeaturesCorrelacionadas = varianza  # Umbral aplicado para descartar features cuya correlacion sea mayor que él
cv_todos = 25  # CROSS_VALIDATION: número de iteraciones. Sirve para evitar el overfitting
fraccion_train = 0.50  # Fracción de datos usada para entrenar
fraccion_test = 0.25  # Fracción de datos usada para testear (no es validación)
fraccion_valid = 1.00 - (fraccion_train + fraccion_test)

print("umbralCasosSuficientesClasePositiva (positivos en pasado de train+test+valid) = " + str(umbralCasosSuficientesClasePositiva))
print("umbralProbTargetTrue = " + str(umbralProbTargetTrue))
print("granProbTargetUno = " + str(granProbTargetUno))

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

print("dir_subgrupo: %s" % dir_subgrupo)
print("modoTiempo: %s" % modoTiempo)
print("desplazamientoAntiguedad: %s" % desplazamientoAntiguedad)
print("pathCsvReducido: %s" % pathCsvReducido)
print("dir_subgrupo_img = %s" % dir_subgrupo_img)
print("umbralFeaturesCorrelacionadas = " + str(umbralFeaturesCorrelacionadas))


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


    # #####
    # # Variables dependientes de SP500
    #
    # # FEATURE:
    # # DEFINICIÓN: compara el rendimiento de la empresa hoy (close - open) respecto al SP500 de ayer
    # # SE QUITA PORQUE NO SE HA VISTO QUE SEA ÚTIL, Y REQUIERE MUCHA COMPUTACIÓN
    # today = date.today()
    # fechaInicio = "2019-01-01"
    # fechaFin = today.strftime("%Y-%m-%d")
    # sp500 = getSP500conRentaTrasXDias(1, fechaInicio, fechaFin, dir_subgrupo)
    # entradaFeaturesYTarget['COMPARA-SP500-AYER-BOOL'] = entradaFeaturesYTarget.apply(
    #     lambda x: anadeComparacionSencillaSP500(sp500, x), axis=1)
    #
    # ######
    #
    # # # Variables complejas, que comparan periodos de la empresa con periodos del SP500
    #
    # ESTAS VARIABLES COMPLEJAS NUNCA SE HAN VALIDADO
    #
    # # Se comparará con la empresa, en varios periodos.Se crearán nuevas variables
    # # Se cogen todas las filas de la primera empresa, para buscar la menor y mayor antigüedad
    # empresa = entradaFeaturesYTarget.iloc[0, 0]
    # filasDeEmpresa = entradaFeaturesYTarget[entradaFeaturesYTarget['empresa'] == empresa]
    # primeraFila = filasDeEmpresa.head(1).reset_index()
    # ultimaFila = filasDeEmpresa.tail(1).reset_index()
    # primeraYUltimaFilas = filasDeEmpresa.iloc[[0, -1]]
    # fechaMasReciente = str(primeraFila.loc[0, 'anio']) + "-" + str(primeraFila.loc[0, 'mes']) + "-" + str(
    #     primeraFila.loc[0, 'dia'])
    # fechaMasAntigua = str(ultimaFila.loc[0, 'anio']) + "-" + str(ultimaFila.loc[0, 'mes']) + "-" + str(
    #     ultimaFila.loc[0, 'dia'])
    #
    # # Se cargan los rangos del SP500 en dicha fecha
    # sp500 = getSP500conRentaTrasXDias(1, fechaMasAntigua, fechaMasReciente, dir_subgrupo)
    #
    # # En el dataframe final, para cada fila se compara la evolución del precio de la empresa vs la evolución del SP500.
    # # IMPORTANTE: el rango del periodo debe estar dentro del rango completo de fechas de la empresa, para no saltar a
    # # otra empresa. Se asume que todas las filas de una empresa están juntas y en orden cronológico, de más reciente
    # # a más antigua.
    # periodos = [1, 3]
    # numFilas = len(entradaFeaturesYTarget.index)
    # for periodo in periodos:
    #     nombreNuevaFeature = "COINCIDE_CON_SP500_" + str(periodo)
    #     for indexActual, row in entradaFeaturesYTarget.iterrows():
    #         indexAntes = indexActual + periodo
    #         if indexAntes < numFilas:
    #             fechaAhora = str(row['anio']) + "-" + str(
    #                 row['mes']) + "-" + str(row['dia'])
    #             fechaAntes = str(entradaFeaturesYTarget.loc[indexAntes, 'anio']) + "-" + str(
    #                 entradaFeaturesYTarget.loc[indexAntes, 'mes']) + "-" + str(
    #                 entradaFeaturesYTarget.loc[indexAntes, 'dia'])
    #             rentaEnPeriodoEmpresa = 100 * (float((row['close']) - float(
    #                 entradaFeaturesYTarget.loc[indexAntes, 'close']))) / float(
    #                 entradaFeaturesYTarget.loc[indexAntes, 'close'])
    #
    #             rentaAhoraSP500=float(sp500[sp500[ETIQUETA_FECHA_SP500] == fechaAhora][ETIQUETA_RENTA_SP500])
    #             rentaAntesSP500 = float(sp500[sp500[ETIQUETA_FECHA_SP500] == fechaAntes][ETIQUETA_RENTA_SP500])
    #             rentaEnPeriodoSP500=rentaAhoraSP500-rentaAntesSP500
    #             # El nombre de la empresa debe coincidir. Si no, no tengo histórico suficiente
    #             if row['empresa'] == entradaFeaturesYTarget.loc[indexAntes, 'empresa']:
    #                 coincide = int((rentaEnPeriodoEmpresa * rentaEnPeriodoSP500) > 0)
    #                 entradaFeaturesYTarget.loc[indexActual, nombreNuevaFeature] = coincide
    #         else:
    #             #serieFeature.add(np.nan)
    #             a=1
    #         c = 1
    #
    #         b = 1


    # ##################### Variables de TWITTER ################
    # # SE QUITAN PORQUE PARA 100 EMPRESAS TARDA 10 MINUTOS POR SUBGRUPO, Y APENAS INFLUENCIA
    #
    # print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " Se inicia el procesado de TWITTER...")
    #
    # # # FEATURE: Número de menciones del ticker por un conjunto de tuiteros que recomiendan a ojo en un
    # # # rango de "antiguedadMaxima" días. Se sumará 1 por cada día y tuitero, si éste lo menciona los últimos x días.
    # # # DEFINICIÓN de FEATURE:
    # # # Para la empresa XXXX, la feature contará cuántos tuiteros mencionan el ticker $XXXX en el rango de días
    # # # El resultado será 0, 1, 2, 3...
    # # entradaFeaturesYTarget=anadeFeatureTwitter(entradaFeaturesYTarget, "MENCIONES-TWITTER-RANGO",
    # #                                                            pathDestinoTweets, tuiterosRecomendacionAOjo,
    # #                                                            4, "")
    #
    # # FEATURE: Número de menciones del ticker por un conjunto de tuiteros que nombran los Premarket Top Gainers.
    # # Se sumará 1 por cada día y tuitero, si éste lo menciona x días antes.
    # # DEFINICIÓN de FEATURE:
    # # Para la empresa XXXX, la feature contará cuántos tuiteros mencionan el ticker $XXXX en el rango de días
    # # El resultado será 0, 1, 2, 3...
    # entradaFeaturesYTarget = anadeFeatureTwitter(entradaFeaturesYTarget, "MENCIONES-TWITTER-GAINERS",
    #                                                              pathDestinoTweets, tuiterosGainers,
    #                                                              3, "Gainers")
    #
    # # FEATURE: Número de menciones del ticker por un conjunto de tuiteros que nombran los Premarket Top Losers.
    # # Se sumará 1 por cada día y tuitero, si éste lo menciona x días antes.
    # # DEFINICIÓN de FEATURE:
    # # Para la empresa XXXX, la feature contará cuántos tuiteros mencionan el ticker $XXXX en el rango de días
    # # El resultado será 0, 1, 2, 3...
    # entradaFeaturesYTarget = anadeFeatureTwitter(entradaFeaturesYTarget, "MENCIONES-TWITTER-LOSERS",
    #                                                              pathDestinoTweets, tuiterosLosers,
    #                                                              3, "Losers")
    #
    # # FEATURE: Número de menciones del ticker por un conjunto de tuiteros que nombran los Most Volatile Gappers.
    # # Se sumará 1 por cada día y tuitero, si éste lo menciona x días antes.
    # # DEFINICIÓN de FEATURE:
    # # Para la empresa XXXX, la feature contará cuántos tuiteros mencionan el ticker $XXXX en el rango de días
    # # El resultado será 0, 1, 2, 3...
    # entradaFeaturesYTarget = anadeFeatureTwitter(entradaFeaturesYTarget, "MENCIONES-TWITTER-GAPPERS",
    #                                              pathDestinoTweets, tuiterosGappers,
    #                                              3, "Gappers")
    #
    # # FEATURE: Número de menciones del ticker por un conjunto de tuiteros que nombran los
    # # "Stocks showing early gains in premarket".
    # # Se sumará 1 por cada día y tuitero, si éste lo menciona x días antes.
    # # DEFINICIÓN de FEATURE:
    # # Para la empresa XXXX, la feature contará cuántos tuiteros mencionan el ticker $XXXX en el rango de días
    # # El resultado será 0, 1, 2, 3...
    # entradaFeaturesYTarget = anadeFeatureTwitter(entradaFeaturesYTarget, "MENCIONES-TWITTER-EARLYGAINSPREMARKET",
    #                                              pathDestinoTweets, tuiterosEarlyGainsPremarket,
    #                                              3, "Stocks showing early gains in premarket")
    #
    # # FEATURE: Número de menciones del ticker por un conjunto de tuiteros que nombran los
    # # "Premarket stock gainers".
    # # Se sumará 1 por cada día y tuitero, si éste lo menciona x días antes.
    # # DEFINICIÓN de FEATURE:
    # # Para la empresa XXXX, la feature contará cuántos tuiteros mencionan el ticker $XXXX en el rango de días
    # # El resultado será 0, 1, 2, 3...
    # entradaFeaturesYTarget = anadeFeatureTwitter(entradaFeaturesYTarget, "MENCIONES-TWITTER-PREMARKETSTOCKGAINERS",
    #                                              pathDestinoTweets, tuiterosPremarketStockGainers,
    #                                              3, "Premarket stock gainers")
    #
    # # FEATURE: Número de menciones del ticker por un conjunto de tuiteros que nombran los
    # # "Stock gainers on watch".
    # # Se sumará 1 por cada día y tuitero, si éste lo menciona x días antes.
    # # DEFINICIÓN de FEATURE:
    # # Para la empresa XXXX, la feature contará cuántos tuiteros mencionan el ticker $XXXX en el rango de días
    # # El resultado será 0, 1, 2, 3...
    # entradaFeaturesYTarget = anadeFeatureTwitter(entradaFeaturesYTarget, "MENCIONES-TWITTER-STOCKGAINERSTOWATCH",
    #                                              pathDestinoTweets, tuiterosStockGainersToWatch,
    #                                              3, "Stock gainers on watch")
    #
    # # FEATURE: Número de menciones del ticker por un conjunto de tuiteros que nombran los
    # # "Today’s Top % Losers".
    # # Se sumará 1 por cada día y tuitero, si éste lo menciona x días antes.
    # # DEFINICIÓN de FEATURE:
    # # Para la empresa XXXX, la feature contará cuántos tuiteros mencionan el ticker $XXXX en el rango de días
    # # El resultado será 0, 1, 2, 3...
    # entradaFeaturesYTarget = anadeFeatureTwitter(entradaFeaturesYTarget, "MENCIONES-TWITTER-TOPLOSERS",
    #                                              pathDestinoTweets, tuiterosTopLosers,
    #                                              3, "Today’s Top % Losers")
    #
    #
    # # FEATURE: Número de menciones del ticker por un conjunto de tuiteros que nombran los
    # # "Today’s Most Active Penny Stocks".
    # # Se sumará 1 por cada día y tuitero, si éste lo menciona x días antes.
    # # DEFINICIÓN de FEATURE:
    # # Para la empresa XXXX, la feature contará cuántos tuiteros mencionan el ticker $XXXX en el rango de días
    # # El resultado será 0, 1, 2, 3...
    # entradaFeaturesYTarget = anadeFeatureTwitter(entradaFeaturesYTarget, "MENCIONES-TWITTER-MOSTACTIVEPENNYSTOCKS",
    #                                              pathDestinoTweets, tuiterosMostActivePennyStocks,
    #                                              3, "Today’s Most Active Penny Stocks")
    #
    #
    # print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " ... se finaliza el procesado de TWITTER")
    #
    # #########################

    ######################
    # Se guarda el conjunto de datos con las nuevas features añadidas en Python. Se usará sólo para depuración
    print("Se guarda el conjunto de features, que incluyen las de Python, en: "+pathCsvCompletoConFeaturesPython)
    entradaFeaturesYTarget.to_csv(pathCsvCompletoConFeaturesPython, index=False, sep='|', float_format='%.4f')
    #################################

    ############# MUY IMPORTANTE: creamos el IDENTIFICADOR UNICO DE FILA, que será el indice!!
    nuevoindice = entradaFeaturesYTarget["empresa"].astype(str) + "_" + entradaFeaturesYTarget["anio"].astype(str) \
                  + "_" + entradaFeaturesYTarget["mes"].astype(str) + "_" + entradaFeaturesYTarget["dia"].astype(str)
    entradaFeaturesYTarget = entradaFeaturesYTarget.reset_index().set_index(nuevoindice, drop=True)
    entradaFeaturesYTarget = entradaFeaturesYTarget.drop('index', axis=1)  # columna index no util ya

    C5C6ManualFunciones.mostrarEmpresaConcreta(entradaFeaturesYTarget, DEBUG_EMPRESA, DEBUG_MES, DEBUG_DIA, 30)
    ###########################################################################

    entradaFeaturesYTarget.sort_index(
        inplace=True)  # Reordenando segun el indice (para facilitar el testIntegracion)
    entradaFeaturesYTarget.to_csv(pathCsvIntermedio + ".entrada", index=True,
                                  sep='|', float_format='%.4f')  # NO BORRAR: UTIL para testIntegracion

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
    print("entradaFeaturesYTarget2 (columnas nulas borradas): " + str(entradaFeaturesYTarget2.shape[0]) + " x " + str(
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

    print("Porcentaje de MISSING VALUES en cada columna del dataframe de entrada (mostramos las que superen " + str(100 * UMBRAL_COLUMNAS_DEMASIADOS_NULOS) + "%):")
    missing = pd.DataFrame(entradaFeaturesYTarget2.isnull().sum()).rename(columns={0: 'total'})
    missing['percent'] = missing['total'] / len(entradaFeaturesYTarget2)  # Create a percentage missing
    missing_df = missing.sort_values('percent', ascending=False)
    missing_df = missing_df[missing_df['percent'] > UMBRAL_COLUMNAS_DEMASIADOS_NULOS]
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
    entradaFeaturesYTarget2 = entradaFeaturesYTarget2.drop('volumen', axis=1).drop('high', axis=1) \
        .drop('low', axis=1).drop('close', axis=1).drop('open', axis=1)

    print("entradaFeaturesYTarget2 (tras quitar columnas dinamicas identificadoras y con valores absolutos): " + str(entradaFeaturesYTarget2.shape[0]) + " x " + str(
        entradaFeaturesYTarget2.shape[1]))

    tablaDebugDF = entradaFeaturesYTarget2.filter(like=DEBUG_FILTRO, axis=0)
    print("tablaDebugDF (caso vigilado) - Tras quitar columnas dinamicas identificadoras y con valores absolutos:")
    print(tabulate(tablaDebugDF, headers='keys', tablefmt='psql'))

    if modoTiempo == "pasado":
        print("Borrar COLUMNAS dinamicas con demasiados nulos (umbral = " + str(
            UMBRAL_COLUMNAS_DEMASIADOS_NULOS) + ")...")
        columnasDemasiadosNulos = missing_df.index.values
        print("columnasDemasiadosNulos: " + columnasDemasiadosNulos)

        print("Guardando columnasDemasiadosNulos en: " + pathColumnasConDemasiadosNulos)
        pickle.dump(columnasDemasiadosNulos, open(pathColumnasConDemasiadosNulos, 'wb'))

        # print(tabulate(entradaFeaturesYTarget2.head(), headers='keys', tablefmt='psql'))
        entradaFeaturesYTarget2 = entradaFeaturesYTarget2.drop(columnasDemasiadosNulos, axis=1)
        print("entradaFeaturesYTarget2: " + str(entradaFeaturesYTarget2.shape[0]) + " x " + str(
            entradaFeaturesYTarget2.shape[1]))

        tablaDebugDF = entradaFeaturesYTarget2.filter(like=DEBUG_FILTRO, axis=0)
        print("tablaDebugDF (caso vigilado) - Tras quitar las columnas con demasiados nulos:")
        print(tabulate(tablaDebugDF, headers='keys', tablefmt='psql'))

        print(
            "MISSING VALUES (FILAS) - Borramos las FILAS que tengan 1 o mas valores NaN porque son huecos que no deberían estar...")
        entradaFeaturesYTarget3 = entradaFeaturesYTarget2.dropna(axis=0, how='any')  # Borrar FILA si ALGUNO sus valores tienen NaN
        print("entradaFeaturesYTarget3: " + str(entradaFeaturesYTarget3.shape[0]) + " x " + str(
            entradaFeaturesYTarget3.shape[1]))

        tablaDebugDF = entradaFeaturesYTarget3.filter(like=DEBUG_FILTRO, axis=0)
        print("tablaDebugDF (caso vigilado) - entradaFeaturesYTarget3:")
        print(tabulate(tablaDebugDF, headers='keys', tablefmt='psql'))

    elif modoTiempo == "futuro":
        print("Borrar COLUMNAS dinamicas con demasiados nulos desde un fichero del pasado..." + pathColumnasConDemasiadosNulos)
        columnasDemasiadosNulos = pickle.load(open(pathColumnasConDemasiadosNulos, 'rb'))
        # print("columnasDemasiadosNulos: "); print(columnasDemasiadosNulos)
        # print(tabulate(entradaFeaturesYTarget2.head(), headers='keys', tablefmt='psql'))
        entradaFeaturesYTarget2 = entradaFeaturesYTarget2.drop(columnasDemasiadosNulos, axis=1)
        print("entradaFeaturesYTarget2: " + str(entradaFeaturesYTarget2.shape[0]) + " x " + str(
            entradaFeaturesYTarget2.shape[1]))

        print(
            "MISSING VALUES (FILAS) - Para el FUTURO; el target es NULO siempre...")
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

    print("entradaFeaturesYTarget3 (filas con algun nulo borradas):" + str(
        entradaFeaturesYTarget3.shape[0]) + " x " + str(entradaFeaturesYTarget3.shape[1]))

    entradaFeaturesYTarget3.to_csv(pathCsvIntermedio + ".sololascompletas.csv", index=True,
                                   sep='|')  # NO BORRAR: UTIL para testIntegracion
    entradaFeaturesYTarget3.to_csv(pathCsvIntermedio + ".sololascompletas_INDICES.csv",
                                   columns=[])  # NO BORRAR: UTIL para testIntegracion

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
    print("Comprobando que el dataframe de entrada tenga datos...")
    num_filas = df3aux.shape[0]
    if (num_filas == 0):
        raise RuntimeError("Hay 0 filas en el dataframe de entrada. Salimos...")

    print("Cargando modelo detector de outliers: " + pathModeloOutliers)
    detector_outliers = pickle.load(open(pathModeloOutliers, 'rb'))
    flagAnomaliasDf = pd.DataFrame(
        {'marca_anomalia': detector_outliers.predict(df3aux)})  # vale -1 es un outlier; si es un 1, no lo es

    if limpiarOutliers is True:
        indice3 = entradaFeaturesYTarget3.index  # lo guardo para pegarlo luego
        entradaFeaturesYTarget3.reset_index(drop=True, inplace=True)
        flagAnomaliasDf.reset_index(drop=True, inplace=True)
        entradaFeaturesYTarget4 = pd.concat([entradaFeaturesYTarget3, flagAnomaliasDf],
                                            axis=1)  # Column Bind, manteniendo el índice del DF izquierdo
        entradaFeaturesYTarget4.set_index(indice3, inplace=True)  # ponemos el indice que tenia el DF de la izquierda

        entradaFeaturesYTarget4 = entradaFeaturesYTarget4.loc[
            entradaFeaturesYTarget4['marca_anomalia'] == 1]  # Cogemos solo las que no son anomalias
        entradaFeaturesYTarget4 = entradaFeaturesYTarget4.drop('marca_anomalia', axis=1)  # Quitamos la columna auxiliar

    else:
        entradaFeaturesYTarget4 = entradaFeaturesYTarget3

    print("entradaFeaturesYTarget4 (sin outliers):" + str(entradaFeaturesYTarget4.shape[0]) + " x " + str(
        entradaFeaturesYTarget4.shape[1]))

    tablaDebugDF = entradaFeaturesYTarget4.filter(like=DEBUG_FILTRO, axis=0)
    print("tablaDebugDF (caso vigilado) - entradaFeaturesYTarget4:")
    print(tabulate(tablaDebugDF, headers='keys', tablefmt='psql'))

    entradaFeaturesYTarget4.to_csv(pathCsvIntermedio + ".sinoutliers.csv", index=True,
                                   sep='|')  # NO BORRAR: UTIL para testIntegracion
    entradaFeaturesYTarget4.to_csv(pathCsvIntermedio + ".sinoutliers_INDICES.csv",
                                   columns=[])  # NO BORRAR: UTIL para testIntegracion

    # ENTRADA: features (+ target)
    if modoTiempo == "pasado":
        featuresFichero = entradaFeaturesYTarget4.drop('TARGET', axis=1)  # default
        targetsFichero = (entradaFeaturesYTarget4[['TARGET']] == 1)  # default
    else:
        featuresFichero = entradaFeaturesYTarget4

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

        print("NO balanceamos clases en capa 5 (pero seguramente sí en capa 6 solo sobre dataset de TRAIN)!!!")
        ift_minoritaria = entradaFeaturesYTarget4[entradaFeaturesYTarget4.TARGET == True]
        ift_mayoritaria = entradaFeaturesYTarget4[entradaFeaturesYTarget4.TARGET == False]
        print("Tasa de desbalanceo entre clases = " + str(ift_mayoritaria.shape[0]) + "/" +
              str(ift_minoritaria.shape[0]) + " = " + str(ift_mayoritaria.shape[0] / ift_minoritaria.shape[0]))
        entradaFeaturesYTarget5 = entradaFeaturesYTarget4

        tablaDebugDF = entradaFeaturesYTarget5.filter(like=DEBUG_FILTRO, axis=0)
        print("tablaDebugDF (caso vigilado) - entradaFeaturesYTarget5:")
        print(tabulate(tablaDebugDF, headers='keys', tablefmt='psql'))

        # entradaFeaturesYTarget5.to_csv(path_csv_completo + "_TEMP05", index=True, sep='|')  # UTIL para testIntegracion
        featuresFichero = entradaFeaturesYTarget5.drop('TARGET', axis=1)
        targetsFichero = entradaFeaturesYTarget5[['TARGET']]
        targetsFichero = (targetsFichero[['TARGET']] == 1)  # Convierto de int a boolean

        print("entradaFeaturesYTarget5 (sin outliers):" + str(entradaFeaturesYTarget5.shape[0]) + " x " + str(
            entradaFeaturesYTarget5.shape[1]))

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
    if evitarNormalizarNiTramificar is False:
        print("Normalizando cada feature...")
        # Vamos a normalizar z-score (media 0, std_dvt=1), pero yeo-johnson tiene un bug (https://github.com/scipy/scipy/issues/10821) que se soluciona sumando una constante a toda la matriz, lo cual no afecta a la matriz normalizada
        featuresFichero = featuresFichero + 1.015815

        if modoTiempo == "pasado":
            # Con el "normalizador COMPLEJO" solucionamos este bug: https://github.com/scikit-learn/scikit-learn/issues/14959  --> Aplicar los cambios indicados a:_/home/carloslinux/Desktop/PROGRAMAS/anaconda3/envs/BolsaPython/lib/python3.7/site-packages/sklearn/preprocessing/_data.py
            modelo_normalizador = make_pipeline(MinMaxScaler(),
                                                PowerTransformer(method='yeo-johnson', standardize=True, copy=True), ).fit(
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
                                   sep='|', float_format='%.4f')  # UTIL para testIntegracion

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

    ##############################################################################
    else:
        # NO NORMALIZAR y NO TRAMIFICAR
        featuresFichero3 = featuresFichero
    ##############################################################################

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
            #print("Usando PCA, creamos una NUEVA BASE DE FEATURES ORTOGONALES y cogemos las que tengan un impacto agregado sobre el "+str(varianza)+"% de la varianza del target. Descartamos el resto.")
            #modelo_pca_subgrupo = PCA(n_components=varianza, svd_solver='full')  # Varianza acumulada sobre el target

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

        print("Dimensiones del dataframe tras PCA: " + str(featuresFichero3_pca.shape[0]) + " x " + str(
            featuresFichero3_pca.shape[1]))

        # print(
        # "Las features están ya normalizadas, reducidas y en base ortogonal PCA. DESCRIBIMOS lo que hemos hecho y GUARDAMOS el dataset.")
        num_columnas_pca = featuresFichero3_pca.shape[1]
        columnas_pca = ["pca_" + f"{i:0>2}" for i in
                        range(num_columnas_pca)]  # Hacemos left padding con la funcion f-strings
        featuresFichero3_pca_df = pd.DataFrame(featuresFichero3_pca, columns=columnas_pca,
                                            index=featuresFichero3.index)
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
        featuresytargets.to_csv(pathCsvReducido, index=True, sep='|', float_format='%.4f')

        ####################################################################

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

    tasaDesbalanceoAntes = round(ift_mayoritaria.shape[0] / ift_minoritaria.shape[0], 2)
    print("Tasa de desbalanceo entre clases (antes de balancear INICIO) = " + str(ift_mayoritaria.shape[0]) + "/" + str(
        ift_minoritaria.shape[0]) + " = " + str(tasaDesbalanceoAntes))
    num_muestras_minoria = ift_minoritaria.shape[0]

    casosInsuficientes = (num_muestras_minoria < umbralCasosSuficientesClasePositiva)
    if (casosInsuficientes):
        print("Numero de casos en clase minoritaria es INSUFICIENTE: " + str(num_muestras_minoria) + " (umbral=" + str(
            umbralCasosSuficientesClasePositiva) + "). Así que abandonamos este dataset y seguimos")

    else:
        print("En el dataframe se coloca primero todas las mayoritarias y despues todas las minoritarias")
        ift_juntas = pd.concat([ift_mayoritaria.reset_index(drop=True), ift_minoritaria.reset_index(drop=True)],
                               axis=0)  # Row bind
        indices_juntos = ift_mayoritaria.index.append(ift_minoritaria.index)  # Row bind
        ift_juntas.set_index(indices_juntos, inplace=True)
        print("Las clases juntas son:")
        print("ift_juntas:" + str(ift_juntas.shape[0]) + " x " + str(ift_juntas.shape[1]))

        ift_juntas.to_csv(pathCsvIntermedio + ".trasbalancearclases.csv", index=True, sep='|',
                          float_format='%.4f')  # NO BORRAR: UTIL para testIntegracion
        ift_juntas.to_csv(pathCsvIntermedio + ".trasbalancearclases_INDICES.csv",
                          columns=[])  # NO BORRAR: UTIL para testIntegracion

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
        upper = matrizCorr.where(np.triu(np.ones(matrizCorr.shape), k=1).astype(bool))
        # print(upper.to_string())
        print("Eliminamos las features muy correladas (umbral = " + str(umbralFeaturesCorrelacionadas) + "):")
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
        # ##################################################################
        columnasSeleccionadas = ift_juntas.columns
        print("Guardando las columnas seleccionadas en: ", pathFeaturesSeleccionadas)
        # print(columnasSeleccionadas)
        columnasSeleccionadasStr = '|'.join(columnasSeleccionadas)
        featuresSeleccionadasFile = open(pathFeaturesSeleccionadas, "w")
        featuresSeleccionadasFile.write(columnasSeleccionadasStr)
        featuresSeleccionadasFile.close()

        ##################################################################

        ############################## DIVISIÓN DE DATOS: TRAIN, TEST, VALIDACIÓN ##########################
        ######## Las filas se randomizan (shuffle) con .sample(frac=1).reset_index(drop=True) #####
        print((datetime.datetime.now()).strftime(
            "%Y%m%d_%H%M%S") + " DIVIDIR EL DATASET DE ENTRADA EN 3 PARTES: TRAIN (" + str(
            fraccion_train) + "), TEST (" + str(fraccion_test) + "), VALIDACION (" + str(
            round(fraccion_valid, 2)) + ")")

        dfBarajado = ift_juntas.sample(frac=1)  # Los indices aparecen bien
        ds_train, ds_test, ds_validacion = np.split(dfBarajado,
                                                    [int(fraccion_train * len(ift_juntas)),
                                                     int((fraccion_train + fraccion_test) * len(ift_juntas))])
        print("TRAIN = " + str(ds_train.shape[0]) + " x " + str(ds_train.shape[1]) + "  " + "TEST --> " + str(
            ds_test.shape[0]) + " x " + str(ds_test.shape[1]) + "  " + "VALIDACION --> " + str(
            ds_validacion.shape[0]) + " x " + str(ds_validacion.shape[1]))

        print("Separamos FEATURES y TARGETS, de los 3 dataframes...")
        ds_train_f = ds_train.drop('TARGET', axis=1).to_numpy()
        ds_train_t = ds_train[['TARGET']].to_numpy().ravel()
        ds_test_f = ds_test.drop('TARGET', axis=1).to_numpy()
        ds_test_t = ds_test[['TARGET']].to_numpy().ravel()
        ds_validac_f = ds_validacion.drop('TARGET', axis=1).to_numpy()
        ds_validac_t = ds_validacion[['TARGET']].to_numpy().ravel()

        feature_names = ds_train.columns.drop('TARGET')

        ################ DESBALANCEOS en train, test y validation ########

        df_mayoritaria = ds_train_t[ds_train_t == False]  # En este caso los mayoritarios son los False
        df_minoritaria = ds_train_t[ds_train_t == True]
        # print("df_mayoritaria (train):" + str(len(df_mayoritaria)))
        # print("df_minoritaria (train):" + str(len(df_minoritaria)))
        tasaDesbalanceoAntes = round(len(df_mayoritaria) / len(df_minoritaria), 2)
        print(
            "TRAIN - Tasa de desbalanceo entre clases (antes de balancear con SMOTE) = mayoritaria/minoritaria = " + str(
                len(df_mayoritaria)) + " / " + str(len(df_minoritaria)) + " = " + str(tasaDesbalanceoAntes))

        df_mayoritaria_test = ds_test_t[ds_test_t == False]  # En este caso los mayoritarios son los False
        df_minoritaria_test = ds_test_t[ds_test_t == True]
        # print("df_mayoritaria (test):" + str(len(df_mayoritaria_test)))
        # print("df_minoritaria (test):" + str(len(df_minoritaria_test)))
        tasaDesbalanceoAntes_test = round(len(df_mayoritaria_test) / len(df_minoritaria_test), 2)
        print("TEST - Tasa de desbalanceo entre clases = mayoritaria/minoritaria = " + str(
            len(df_mayoritaria_test)) + " / " + str(len(df_minoritaria_test)) + " = " + str(tasaDesbalanceoAntes_test))

        df_mayoritaria_validac = ds_validac_t[ds_validac_t == False]  # En este caso los mayoritarios son los False
        df_minoritaria_validac = ds_validac_t[ds_validac_t == True]
        # print("df_mayoritaria (validac):" + str(len(df_mayoritaria_validac)))
        # print("df_minoritaria (validac):" + str(len(df_minoritaria_validac)))
        tasaDesbalanceoAntes_validac = round(len(df_mayoritaria_validac) / len(df_minoritaria_validac), 2)
        print("VALIDAC - Tasa de desbalanceo entre clases = mayoritaria/minoritaria = " + str(
            len(df_mayoritaria_validac)) + " / " + str(len(df_minoritaria_validac)) + " = " + str(
            tasaDesbalanceoAntes_validac))

        ########################### SMOTE (Over/Under sampling) ##################
        ########################### Se aplica sólo en el TRAIN #############################################3

        balancearConSmoteSoloTrain = balancearConSmoteSoloTrain and (
                tasaDesbalanceoAntes > umbralNecesarioCompensarDesbalanceo)
        balancearUsandoDownsampling = balancearUsandoDownsampling and (
                tasaDesbalanceoAntes > umbralNecesarioCompensarDesbalanceo)
        ds_train_sinsmote = ds_train  # NO TOCAR
        ds_train_f_sinsmote = ds_train_f  # NO TOCAR
        ds_train_t_sinsmote = ds_train_t  # NO TOCAR
        columnas_f = ds_train_sinsmote.drop('TARGET', axis=1).columns
        if modoTiempo == "pasado" and balancearConSmoteSoloTrain:
            print((datetime.datetime.now()).strftime(
                "%Y%m%d_%H%M%S") + " ---------------- RESAMPLING con SMOTE (y porque supera umbral = " + str(
                umbralNecesarioCompensarDesbalanceo) + ") --------")
            print("Resampling con SMOTE del vector de TRAINING (pero no a TEST ni a VALIDATION) según: "
                  + "https://machinelearningmastery.com/combine-oversampling-and-undersampling-for-imbalanced-classification/")
            resampleST = SMOTETomek()
            print("SMOTE antes (mayoritaria + minoritaria): %d" % ds_train_f_sinsmote.shape[0])
            print("SMOTE fit...")
            ds_train_f, ds_train_t = resampleST.fit_resample(ds_train_f_sinsmote, ds_train_t_sinsmote)
            ds_train_f = pd.DataFrame(ds_train_f, columns=columnas_f)  # restablecer nombres de columnas
            ds_train_t = pd.DataFrame(ds_train_t, columns=['TARGET'])  # restablecer nombres de columnas
            print("SMOTE después (mayoritaria + minoritaria): %d" % ds_train_f.shape[0])

        elif modoTiempo == "pasado" and balancearUsandoDownsampling:
            print((datetime.datetime.now()).strftime(
                "%Y%m%d_%H%M%S") + " ---------------- RESAMPLING haciendo DOWNSAMPLING de la mayoritaria (y porque supera umbral = " + str(
                umbralNecesarioCompensarDesbalanceo) + ") --------")
            print("URL: https://elitedatascience.com/imbalanced-classes")
            print("Solo actua sobre pasado-train, pero no sobre: pasado-test, pasado-valid ni futuro")
            # Separate majority and minority classes
            df_majority = ds_train_sinsmote[ds_train_sinsmote["TARGET"] == False]
            df_minority = ds_train_sinsmote[ds_train_sinsmote["TARGET"] == True]
            # Downsample majority class

            df_majority_downsampled = resample(df_majority,
                                               replace=False,  # sample without replacement
                                               n_samples=len(df_minority),  # to match minority class
                                               random_state=123)  # reproducible results
            df_downsampled = pd.concat([df_majority_downsampled, df_minority])  # junta mayoritaria y minoritaria
            df_downsampled = sklearn.utils.shuffle(df_downsampled)  # barajar
            ds_train_f = pd.DataFrame(df_downsampled.drop('TARGET', axis=1), columns=columnas_f)
            ds_train_t = pd.DataFrame(df_downsampled[['TARGET']], columns=['TARGET'])

        ift_mayoritaria_entrada_modelos = ds_train_t[ds_train_t == False]  # En este caso los mayoritarios son los False
        ift_minoritaria_entrada_modelos = ds_train_t[ds_train_t == True]
        print("ift_mayoritaria_entrada_modelos:" + str(len(ift_mayoritaria_entrada_modelos)))
        print("ift_minoritaria_entrada_modelos:" + str(len(ift_minoritaria_entrada_modelos)))
        tasaDesbalanceoDespues = len(ift_mayoritaria_entrada_modelos) / len(ift_minoritaria_entrada_modelos)
        print("Tasa de desbalanceo entre clases (entrada a los modelos predictivos) = " + str(tasaDesbalanceoDespues))

        ###############################  FIN DE SMOTE ##############################

        ################################# GENERACIÓN DE MODELOS #################################

        # ########################### MODELO SVC balanceado ##################################
        # print("Inicio de SVC")
        # modelo = SVC(class_weight='balanced',  # penalize
        #             probability=True)
        #
        # modelo=modelo.fit(ds_train_f, ds_train_t)
        # print("Fin de SVC")
        # nombreModelo = "svc"
        # ###########################################################################

        # ########################### MODELO MLP ##################################
        # # Resultado: 0.43
        # print("Inicio de MLP")
        # modelo = MLPClassifier()
        #
        # modelo=modelo.fit(ds_train_f, ds_train_t)
        # print("Fin de MLP")
        # nombreModelo = "mlp"
        # ###########################################################################

        # ########################## INICIO DE GRID SEARCH #########################################################
        #
        # print("Inicio de GRID SEARCH")
        #
        # skf = StratifiedKFold(n_splits=10)
        # clf = RandomForestClassifier(n_jobs=-1)
        # param_grid = {
        #     'min_samples_split': [3, 5, 10],
        #     'n_estimators': [100, 200],
        #     'max_depth': [3, 5, 15],
        #     'max_features': [3, 5, 10]
        # }
        # scorers = {
        #     'precision_score': make_scorer(precision_score),
        #     'recall_score': make_scorer(recall_score),
        #     'accuracy_score': make_scorer(accuracy_score)
        # }
        # grid = GridSearchCV(clf, param_grid=param_grid, scoring=scorers, refit='precision_score',
        #                     cv=skf, return_train_score=True, n_jobs=-1)
        #
        # # fit ensemble model to training data
        # modelo = grid.fit(ds_train_f, ds_train_t)  # test our model on the test data
        # nombreModelo = "gridsearch"
        #
        # print("Fin de GRID SEARCH")
        #
        # ########################## FIN DE GRID SEARCH #########################################################

        ########################## INICIO DE ENSEMBLE #########################################################

        # Se calcula ENSEMBLE (mezcla de modelos)
        # print("Inicio de Ensemble")
        # clf1 = SVC(class_weight='balanced',  # penalize
        #              probability=True)
        # #clf2 = RandomForestClassifier(class_weight='balanced') # Rendimiento: 0.45
        # clf2 = MLPClassifier()  # Rendimiento: 0.46
        # eclf = VotingClassifier(estimators=[('m1', clf1), ('m2', clf2)],
        #                 voting='soft')
        # params = {}
        # grid = GridSearchCV(estimator=eclf, param_grid=params, cv=cv_todos, scoring='precision', n_jobs=-1)
        #
        # # fit ensemble model to training data
        # modelo = grid.fit(ds_train_f, ds_train_t)  # test our model on the test data
        # nombreModelo = "ensemble"

        # print("Fin de Ensemble")

        ########################## FIN DE ENSEMBLE #########################################################

        # # ########################## INICIO DE XGBOOST OPTIMIZADO ########################################################33
        # #
        # #
        # #     #################### OPTIMIZACION DE PARAMETROS DE XGBOOST ###############################################################
        #
        # # Parametros por defecto de los modelos que usan árboles de decisión
        # ARBOLES_n_estimators = 80
        # ARBOLES_max_depth = 11
        # ARBOLES_min_samples_leaf = 20
        # ARBOLES_max_features = "auto"
        # ARBOLES_min_samples_split = 3
        # ARBOLES_max_leaf_nodes = None
        # ARBOLES_min_impurity_decrease = 0.001
        #
        # seed = 112  # Random seed
        #
        # # Descomentar para obtener los parámetros con optimización Bayesiana
        # # IMPORTANTE: se debe instalar el paquete de bayes en Conda: conda install -c conda-forge bayesian-optimization
        # # Se imprimirán en el log, pero debo luego meterlos manualmente en el modelo
        # # IMPORTANTE: DEBEN RELLENARSE 2 VALORES POR CADA ATRIBUTO DE PBOUND
        # # https://ayguno.github.io/curious/portfolio/bayesian_optimization.html
        #
        # print("Inicio del optimizador de parametros de XGBOOST...")
        #
        # # Parametros ordenados ALFABETICAMENTE porque la liberia lo obliga
        # pbounds = {
        #     'colsample_bytree': (0.1, 0.4),
        #     'gamma': (2, 10),
        #     'learning_rate': (0.2, 0.4),
        #     'max_delta_step': (0, 10),
        #     'max_depth': (4, 7),
        #     'min_child_weight': (2, 20),
        #     'n_estimators': (10, 50),
        #     'reg_alpha': (0.1, 0.8)
        # }
        #
        # hyperparameter_space = {
        # }
        #
        #
        # def xgboost_hyper_param(max_depth, learning_rate, n_estimators, reg_alpha, min_child_weight, colsample_bytree,
        #                         gamma, max_delta_step):
        #     """Crea un modelo XGBOOST con los parametros indicados en la entrada. Aplica el numero de iteraciones de cross-validation indicado
        #         """
        #     clf = XGBClassifier(colsample_bytree=colsample_bytree, gamma=gamma, learning_rate=learning_rate,
        #                         max_depth=int(max_depth), min_child_weight=int(min_child_weight),
        #                         n_estimators=int(n_estimators), reg_alpha=reg_alpha,
        #                         nthread=-1, objective='binary:logistic', seed=seed, use_label_encoder=False,
        #                         eval_metric=["map"], max_delta_step=max_delta_step, scale_pos_weight=1) #'logloss'
        #
        #     # Explicacion: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        #     return np.mean(cross_val_score(clf, ds_train_f, ds_train_t, cv=cv_todos, scoring='accuracy'))
        #
        #
        # # alpha is a parameter for the gaussian process
        # # Note that this is itself a hyperparameter that can be optimized.
        # gp_params = {"alpha": 1e-7}
        #
        # # LIBRERIA: https://github.com/fmfn/BayesianOptimization
        # # Parametros: https://github.com/fmfn/BayesianOptimization/blob/master/bayes_opt/bayesian_optimization.py
        # # Añadir carpeta dinámicamente: https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
        # sys.path.append('/bayes_opt')
        # from bayes_opt import BayesianOptimization
        #
        # optimizer = BayesianOptimization(f=xgboost_hyper_param, pbounds=pbounds, random_state=1,
        #                                  verbose=10)
        #
        # # Fichero de log JSON con los escenarios probados
        # # optimizacion_bayesiana_escenarios = "./optimiz_bayes_escenarios.json"
        # # bo_logger = JSONLogger(path=optimizacion_bayesiana_escenarios)
        # # optimizer.subscribe(Events.OPTIMIZATION_STEP, bo_logger)
        # # if os.path.isfile(optimizacion_bayesiana_escenarios):  # Si ya existe una lista de puntos previa, los precargo
        # #     load_logs(optimizer, logs=[optimizacion_bayesiana_escenarios]);
        # #     print("New optimizer is now aware of {} points.".format(len(optimizer.space)))
        #
        # # print("Optimización de procesos bayesianos - Añadimos ESCENARIOS CONCRETOS para fijarlos (tuplas de parametros) que hayamos visto que tienen buenos resultados...")
        # # optimizer.probe(params={"colsample_bytree": 0.4, "gamma": 2.0, "learning_rate": 0.4, "max_delta_step": 9.6, "max_depth": 7.0, "min_child_weight": 8.2, "n_estimators": 47, "reg_alpha": 0.1}, lazy=False)
        #
        # optimizer.maximize(init_points=5, n_iter=20, acq='ucb', kappa=30, **gp_params)
        # #optimizer.maximize(init_points=5, n_iter=20, acq='poi', kappa=3, **gp_params)
        # # KAPPA: Parameter to indicate how closed are the next parameters sampled
        #
        # valoresOptimizados = optimizer.max
        # print(valoresOptimizados)
        # print("Fin del optimizador")
        # ###################################################################################
        #
        # print("Inicio de XGBOOST")
        # nombreModelo = "xgboost"
        # pathModelo = dir_subgrupo + nombreModelo + ".modelo"
        # # Parametros: https://xgboost.readthedocs.io/en/latest/parameter.html
        # # Instalación en Conda: conda install -c anaconda py-xgboost
        # # Instalación en Python básico: pip install xgboost
        #
        # # MODELO LUIS AUTOOPTIMIZADO PARA CADA SUBGRUPO
        # max_depth = int(valoresOptimizados.get("params").get("max_depth"))
        # learning_rate = valoresOptimizados.get("params").get("learning_rate")
        # n_estimators = int(valoresOptimizados.get("params").get("n_estimators"))
        # reg_alpha = valoresOptimizados.get("params").get("reg_alpha")
        # min_child_weight = int(valoresOptimizados.get("params").get("min_child_weight"))
        # colsample_bytree = valoresOptimizados.get("params").get("colsample_bytree")
        # gamma = valoresOptimizados.get("params").get("gamma")
        # max_delta_step = valoresOptimizados.get("params").get("max_delta_step")
        # nthread = -1
        # objective = 'binary:logistic'
        # seed = seed
        #
        # modelo = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators,
        #                        reg_alpha=reg_alpha, min_child_weight=min_child_weight,
        #                        colsample_bytree=colsample_bytree, gamma=gamma,
        #                        nthread=nthread, objective=objective, seed=seed, use_label_encoder=False,
        #                        max_delta_step=max_delta_step, scale_pos_weight=1)
        #
        # eval_set = [(ds_train_f.to_numpy(), ds_train_t.to_numpy().ravel()), (ds_test_f, ds_test_t)]
        # modelo = modelo.fit(ds_train_f.to_numpy(), ds_train_t.to_numpy().ravel(), eval_metric=["map"],
        #                     early_stopping_rounds=3, eval_set=eval_set,
        #                     verbose=False)  # ENTRENAMIENTO (TRAIN)
        #
        # # ########################## FIN DE XGBOOST OPTIMIZADO ########################################################

        ###################### MODELO LGBM ######################
        from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold, cross_validate, StratifiedShuffleSplit
        from sklearn import metrics
        import lightgbm as lgb

        nombreModelo = "lgbm"

        params = {'objective': 'binary',
                  'learning_rate': 0.08,
                  "boosting_type": "gbdt",
                  "metric": 'precision',
                  'n_jobs': -1,
                  'min_data_in_leaf': 6,
                  'min_child_samples': 3,
                  'num_leaves': 15,  # maximo numero de hojas
                  'max_depth': 4,
                  'random_state': 0,
                  'importance_type': 'split',
                  'min_split_gain': 0.0,
                  # min_child_weight = 0.001,  subsample = 1.0, subsample_freq = 0, colsample_bytree = 1.0, reg_alpha = 0.0, reg_lambda = 0.0,
                  }

        modelo = lgb.LGBMClassifier(**params, n_estimators=50)
        modelo.fit(ds_train_f, ds_train_t, eval_set=[(ds_train_f, ds_train_t), (ds_test_f, ds_test_t)])
        #####################################################

        # ########################## INICIO DE XGBOOST SIN OPTIMIZAR ########################################################
        #
        # nombreModelo = "xgboost_noopt"
        #
        # modelo = XGBClassifier()
        #
        # eval_set = [(ds_train_f, ds_train_t), (ds_test_f, ds_test_t)]
        # modelo = modelo.fit(ds_train_f, ds_train_t, eval_metric=["map"], early_stopping_rounds=4, eval_set=eval_set,
        #                     verbose=False)  # ENTRENAMIENTO (TRAIN)
        #
        # ########################## FIN DE XGBOOST SIN OPTIMIZAR ########################################################

        ############################## GUARDADO DE MODELO #########################
        pathModelo = dir_subgrupo + nombreModelo + ".modelo"
        pickle.dump(modelo, open(pathModelo, 'wb'))

        ############################## APERTURA DE MODELO #########################
        modelo_loaded = pickle.load(open(pathModelo, 'rb'))

        ############################## PREDICCION y CALCULO DE LAS METRICAS EN TEST Y VALID ####################
        print("Inicio de ANÁLISIS DE RESULTADOS - train vs test vs validación")

        print("ds_train_f_sinsmote: " + str(ds_train_f_sinsmote.shape[0]) + " x " + str(ds_train_f_sinsmote.shape[1]))
        train_t_predicho = modelo_loaded.predict(ds_train_f_sinsmote)
        precision_train = precision_score(ds_train_t_sinsmote, train_t_predicho)

        path_dibujo_probabs = dir_subgrupo_img + "histograma_target_probabilidades.png"
        print("Distribución de las probabilidades del target predicho (debe ser con forma de U para que distinga bien los positivos de los negativos): " + path_dibujo_probabs)
        probsPositivosyNegativos = modelo_loaded.predict_proba(ds_train_f_sinsmote)

        # probabilities
        probs = pd.DataFrame(data=probsPositivosyNegativos)
        probs.columns = ['proba_false', 'proba_true']
        print("PASADO - Ejemplos de probabilidades al predecir targets true (orden descendiente): ")
        print(tabulate(probs.sort_values("proba_true", ascending=False).head(n=5), headers='keys', tablefmt='psql'))
        print("PASADO - Ejemplos de probabilidades al predecir targets true (orden ascendiente): ")
        print(tabulate(probs.sort_values("proba_true", ascending=True).head(n=5), headers='keys', tablefmt='psql'))

        probsPositivosyNegativosDF = pd.DataFrame(data=probsPositivosyNegativos, columns=["probabsNegativas", "probabsPositivas"])
        plt.hist(probsPositivosyNegativosDF.iloc[:, 1], label="Probabilidades target predicho", bins=20, alpha=.7, color='blue')
        plt.title("Distribución de la probab predicha al predecir el target (subgrupo " + id_subgrupo + ")", fontsize=10)
        plt.legend(loc='upper left')
        plt.savefig(path_dibujo_probabs, bbox_inches='tight')
        plt.clf(); plt.cla(); plt.close()  # Limpiando dibujo

        print("Informe de metricas:")
        print(classification_report(ds_train_t_sinsmote, train_t_predicho))

        if precision_train == 0:
            # print(train_t_predicho)
            raise NameError(
                "La precision calculada es 0 porque posiblemente no tenemos casos positivos en la muestra predicha. Salimos del proceso de este subgrupo " + id_subgrupo + "...")

        print("PRECISIÓN EN TRAIN = " + '{0:0.2f}'.format(precision_train))
        pd.DataFrame(ds_train_t).to_csv(pathCsvIntermedio + ".ds_train_t_sinsmote.csv", index=True, sep='|',
                                        float_format='%.4f')  # NO BORRAR: UTIL para testIntegracion
        pd.DataFrame(train_t_predicho).to_csv(pathCsvIntermedio + ".train_t_predicho.csv", index=True, sep='|',
                                              float_format='%.4f')  # NO BORRAR: UTIL para testIntegracion

        print("ds_validac_f: " + str(ds_validac_f.shape[0]) + " x " + str(ds_validac_f.shape[1]))
        print("ds_validac_f: " + str(ds_validac_f.shape[0]) + " x " + str(ds_validac_f.shape[1]))
        ds_test_t_pred = modelo_loaded.predict(ds_test_f)
        test_t_predicho = ds_test_t_pred
        validac_t_predicho = modelo_loaded.predict(ds_validac_f)
        precision_test = precision_score(ds_test_t, test_t_predicho)
        precision_avg_test = average_precision_score(ds_test_t, test_t_predicho)
        precision_validation = precision_score(ds_validac_t, validac_t_predicho)
        precision_avg_validation = average_precision_score(ds_validac_t, validac_t_predicho)
        precision_media = (precision_test + precision_validation) / 2
        precision_avg_media = (precision_avg_test + precision_avg_validation) / 2
        print("PRECISIÓN EN TEST = " + '{0:0.2f}'.format(precision_test))
        print("PRECISIÓN EN VALIDACION = " + '{0:0.2f}'.format(precision_validation))


        # NO BORRAR: UTIL para informe HTML entregable
        precisionSistemaRandom = 1 / tasaDesbalanceoAntes # Precision del sistema tonto
        precisionMediaTestValid = (precision_test + precision_validation) / 2
        mejoraRespectoSistemaRandom = 100 * (precisionMediaTestValid - precisionSistemaRandom) / precisionSistemaRandom
        print("ENTREGABLEPRECISIONESPASADO"
              + "|id_subgrupo:" + str(id_subgrupo)
              + "|precisionpasadotrain:" + str(round(precision_train, 2))
              + "|precisionpasadotest:" + str(round(precision_test, 2))
              + "|precisionpasadovalidacion:" + str(round(precision_validation, 2))
              + "|precisionsistemarandom:" + str(round(precisionSistemaRandom, 2))
              + "|mejoraRespectoSistemaRandom:" + str(round(mejoraRespectoSistemaRandom, 0))+" %"
              )

        # NO BORRAR: UTIL para testIntegracion
        pd.DataFrame(ds_test_t).to_csv(pathCsvIntermedio + ".ds_test_t.csv", index=True, sep='|', float_format='%.4f')
        pd.DataFrame(test_t_predicho).to_csv(pathCsvIntermedio + ".test_t_predicho.csv", index=True, sep='|',
                                             float_format='%.4f')
        pd.DataFrame(ds_validac_t).to_csv(pathCsvIntermedio + ".ds_validac_t.csv", index=True, sep='|',
                                          float_format='%.4f')
        pd.DataFrame(validac_t_predicho).to_csv(pathCsvIntermedio + ".validac_t_predicho.csv", index=True, sep='|',
                                                float_format='%.4f')

        print(id_subgrupo + " " + nombreModelo + " -> Precision = " + '{0:0.2f}'.format(
            precision_media) + " (average precision = " + '{0:0.2f}'.format(precision_avg_media) + ")")
        if precision_media > ganador_metrica:
            ganador_metrica = precision_media
            ganador_metrica_avg = precision_avg_media
            ganador_nombreModelo = nombreModelo
            ganador_grid_mejores_parametros = []  # Este modelo no tiene esta variable
        print("Fin de ANÁLISIS DE RESULTADOS")

        ######################################################################################################################
        ######################################################################################################################

        print("********* GANADOR de subgrupo *************")
        num_positivos_test = len(ds_test_t[ds_test_t is True])
        num_positivos_validac = len(ds_validac_t[ds_validac_t is True])
        print("PASADO -> " + id_subgrupo + " (num features = " + str(
            ds_train_f.shape[1]) + ")" + " -> Modelo ganador = " + ganador_nombreModelo + " --> METRICA = " + str(
            round(ganador_metrica, 4)) + " (avg_precision = " + str(round(ganador_metrica_avg, 4)) + ")")

        print("PASADO -> " + id_subgrupo + " TASA DE MEJORA DE PRECISION RESPECTO A RANDOM: ",
              round(ganador_metrica / (1/(1+tasaDesbalanceoAntes)), 2))

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
            df_train_f_sinsmote = pd.DataFrame(ds_train_f_sinsmote, columns=ds_train_f_temp.columns,
                                               index=ds_train_f_temp.index)  # Indice que tenia antes de SMOTE

            C5C6ManualFunciones.comprobarPrecisionManualmente(ds_train_t_sinsmote, train_t_predicho, "TRAIN (forzado)", id_subgrupo,
                                                              df_train_f_sinsmote, dir_subgrupo, DEBUG_FILTRO)  # ds_train_t tiene SMOTE!!!
            C5C6ManualFunciones.comprobarPrecisionManualmente(ds_test_t, test_t_predicho, "TEST", id_subgrupo, df_test_empresas_index,
                                          dir_subgrupo, DEBUG_FILTRO)
            C5C6ManualFunciones.comprobarPrecisionManualmente(ds_validac_t, validac_t_predicho, "VALIDACION", id_subgrupo,
                                          df_valid_empresas_index, dir_subgrupo, DEBUG_FILTRO)
            ######################################################################################################################
        else:
            print("No se ha guardado modelo:" + pathModeloGanadorDeSubgrupoOrigen)

elif (modoTiempo == "futuro" and pathCsvReducido.endswith('.csv') and os.path.isfile(pathCsvReducido) and os.stat(
        pathCsvReducido).st_size > 0):

    print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " Capa 6 - Modo futuro")
    inputFeaturesyTarget = pd.read_csv(pathCsvReducido, index_col=0, sep='|')
    print("inputFeaturesyTarget: " + str(inputFeaturesyTarget.shape[0]) + " x " + str(inputFeaturesyTarget.shape[1]))

    print("Si las hay, eliminamos las features muy correladas (umbral =" + str(
        umbralFeaturesCorrelacionadas) + ") aprendido en el PASADO.")
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

        print("Prediciendo...")
        targets_predichos = modelo_predictor_ganador.predict(inputFeatures_sinnulos.to_numpy())
        num_targets_predichos = len(targets_predichos)
        print("Numero de targets_predichos: " + str(num_targets_predichos) + " con numero de TRUEs = " + str(
            np.sum(targets_predichos, where=["True"])))
        # print("El array de targets contiene:")
        # print(targets_predichos)

        # probabilities
        probs = pd.DataFrame(data=modelo_predictor_ganador.predict_proba(inputFeatures_sinnulos),
                             index=inputFeatures_sinnulos.index)
        probs.columns = ['proba_false', 'proba_true']
        print("FUTURO - Ejemplos de probabilidades al predecir targets true (orden descendiente): ")
        print(tabulate(probs.sort_values("proba_true", ascending=False).head(n=5), headers='keys', tablefmt='psql'))

        # UMBRAL MENOS PROBABLES CON TARGET=1. Cuando el target es 1, se guarda su probabilidad
        # print("El DF llamado probs contiene las probabilidades de predecir un 0 o un 1:")
        # print(probs)
        print("Sobre los targets predichos, se cogen solo aquellos con probabilidad encima del umbral -> umbralProbTargetTrue=" + str(umbralProbTargetTrue) + " y granProbTargetUno=" + str(granProbTargetUno))
        probabilidadesEnTargetUnoPeq = probs.iloc[:, 1]  # Cogemos solo la segunda columna: prob de que sea target=1
        probabilidadesEnTargetUnoPeq2 = probabilidadesEnTargetUnoPeq.apply(
            lambda x: x if (x >= umbralProbTargetTrue) else np.nan)  # Cogemos solo las filas cuya prob_1 > umbral
        probabilidadesEnTargetUnoPeq3 = probabilidadesEnTargetUnoPeq2[
            np.isnan(probabilidadesEnTargetUnoPeq2[:]) == False]  # Cogemos todos los no nulos (NAN)
        # print("El DF llamado probabilidadesEnTargetUnoPeq3 contiene las probabilidades de los UNO con prob mayor que umbral ("+str(umbralProbTargetTrue)+"):")
        # print(probabilidadesEnTargetUnoPeq3)

        probabilidadesEnTargetUnoPeq4 = probabilidadesEnTargetUnoPeq3.sort_values(ascending=False)  # descendente
        # print("El DF llamado probabilidadesEnTargetUnoPeq4 contiene los indices y probabilidades, tras aplicar umbral INFERIOR: " + str(umbralProbTargetTrue) + ". Son:")
        # print(probabilidadesEnTargetUnoPeq4)

        numfilasSeleccionadas = int(granProbTargetUno * probabilidadesEnTargetUnoPeq4.shape[
            0] / 100)  # Como están ordenadas en descendente, cojo estas NUM primeras filas
        print("numfilasSeleccionadas: " + str(numfilasSeleccionadas))
        targets_predichosCorregidos_probs = probabilidadesEnTargetUnoPeq4[0:numfilasSeleccionadas]
        targets_predichosCorregidos = targets_predichosCorregidos_probs.apply(lambda x: 1)
        # print("El DF llamado targets_predichosCorregidos contiene los indices y probabilidades, tras aplicar umbral SUPERIOR: top " + str(granProbTargetUno) + " % de muestras. Son:")
        # print(targets_predichosCorregidos)

        print("Guardando targets PREDICHOS en: " + pathCsvPredichos)
        df_predichos = targets_predichosCorregidos.to_frame()
        df_predichos.columns = ['TARGET_PREDICHO']
        df_predichos.to_csv(pathCsvPredichos, index=False, sep='|',
                            float_format='%.4f')  # Capa 6 - Salida (para el validador, sin indice)

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

        # Predichos con columnas: empresa anio mes dia
        indiceDFPredichos = df_predichos.index.values
        df_predichos.insert(0, column="indiceColumna", value=indiceDFPredichos)

        if df_predichos.shape[0] <= 0:
            print("No se ha predicho NINGUN target=1 sabiendo que cogemos el top " + str(granProbTargetUno) + "% elementos que han tenido probabilidad >= "+ str(umbralProbTargetTrue) + " Por tanto, sa acaba el proceso.")
        else:
            df_predichos[['empresa', 'anio', 'mes', 'dia']] = df_predichos['indiceColumna'].str.split('_', 4, expand=True)
            df_predichos = df_predichos.drop('indiceColumna', axis=1)
            df_predichos = df_predichos.astype({"anio": int, "mes": int, "dia": int})

            indiceDFPredichos = df_predichos_probs.index.values
            df_predichos_probs.insert(0, column="indiceColumna", value=indiceDFPredichos)
            df_predichos_probs[['empresa', 'anio', 'mes', 'dia']] = df_predichos_probs['indiceColumna'].str.split('_', 4,
                                                                                                                  expand=True)
            df_predichos_probs = df_predichos_probs.drop('indiceColumna', axis=1)
            df_predichos_probs = df_predichos_probs.astype({"anio": int, "mes": int, "dia": int})

            print("Juntar COMPLETO con TARGETS PREDICHOS... ")
            df_juntos_1 = pd.merge(df_completo, df_predichos, on=["empresa", "anio", "mes", "dia"], how='left')
            df_juntos_2 = pd.merge(df_juntos_1, df_predichos_probs, on=["empresa", "anio", "mes", "dia"], how='left')

            df_juntos_2['TARGET_PREDICHO'] = (df_juntos_2['TARGET_PREDICHO'] * 1).astype(
                'Int64')  # Convertir de boolean a int64, manteniendo los nulos

            print("Guardando (pathCsvFinalFuturo): " + pathCsvFinalFuturo)
            df_juntos_2.to_csv(pathCsvFinalFuturo, index=False, sep='|', float_format='%.4f')


    else:
        print(
            "No existe el modelo predictor del pasado que necesitamos (" + path_modelo_predictor_ganador + "). Por tanto, no predecimos.")


else:
    print("Los parametros de entrada son incorrectos o el CSV no existe o esta vacio!!")

############################################################
print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " ------------ FIN de capa 6----------------")
