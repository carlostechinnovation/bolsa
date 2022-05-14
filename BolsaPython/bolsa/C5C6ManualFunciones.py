import numpy as np
import pandas as pd
from tabulate import tabulate
import os
import seaborn as sns
import matplotlib.pyplot as plt

def mostrarEmpresaConcreta (miDF, DEBUG_EMPRESA, DEBUG_MES, DEBUG_DIA, numFilasMax):
    tablaDebugDF = miDF[
        (miDF['empresa'] == DEBUG_EMPRESA) & (miDF['mes'] == DEBUG_MES) & (
                    miDF['dia'] == DEBUG_DIA)].head(n=numFilasMax)
    print("tablaDebugDF (caso vigilado) - ENTRADA:")
    print(tabulate(tablaDebugDF, headers='keys', tablefmt='psql'))

def mostrarEmpresaConcretaConFilter (miDF, DEBUG_FILTRO, etiqueta):
    tablaDebugDF = miDF.filter(like=DEBUG_FILTRO, axis=0)
    print("tablaDebugDF (caso vigilado) - " + etiqueta + ":")
    print(tabulate(tablaDebugDF, headers='keys', tablefmt='psql'))

######################### PARÁMETROS TWITTER ########################
import tweepy  # https://github.com/tweepy/tweepy
import csv


pathClavesTwitter = "/bolsa/twitterapikeys/keys"
pathDestinoTweets = "/bolsa/twitter"

FORMATO_FECHA_TWITTER = '%Y-%m-%d'

tuiterosRecomendacionAOjo = ["BagholderQuotes", "vintweeta", "Xiuying"]
tuiterosGainers=["GetScanz"]
tuiterosLosers=["GetScanz"]
tuiterosGappers=["GetScanz"]
tuiterosEarlyGainsPremarket=["FloatChecker"]
tuiterosPremarketStockGainers=["FloatChecker"]
tuiterosStockGainersToWatch=["FloatChecker"]
tuiterosTopLosers=["GetScanz"]
tuiterosMostActivePennyStocks=["GetScanz"]

#####################################################

# PRECISION: de los pocos casos que predigamos TRUE, queremos que todos sean aciertos.
def comprobarPrecisionManualmente(targetsNdArray1, targetsNdArray2, etiqueta, id_subgrupo, dfConIndex, dir_subgrupo, DEBUG_FILTRO):
    print(id_subgrupo + " " + etiqueta + " Comprobación de la precisión --> dfConIndex: " + str(
        dfConIndex.shape[0]) + " x " + str(
        dfConIndex.shape[1]) + ". Se comparan: array1=" + str(targetsNdArray1.size) + " y array2=" + str(
        targetsNdArray2.size))

    df1 = pd.DataFrame(targetsNdArray1, columns=['target'])
    df1.index = dfConIndex.index  # fijamos el indice del DF grande
    df2 = pd.DataFrame(targetsNdArray2, columns=['target'])
    df2.index = dfConIndex.index  # fijamos el indice del DF grande

    df1['targetpredicho'] = df2['target']
    df1['iguales'] = np.where(df1['target'] == df1['targetpredicho'], True, False)

    # Solo nos interesan los PREDICHOS True, porque es donde pondremos dinero real
    df1a = df1[df1.target == True];
    df1b = df1a[df1a.iguales == True]  # Solo los True Positives
    df2a = df1[df1.targetpredicho == True]  # donde ponemos el dinero real (True Positives y False Positives)

    print(etiqueta, " - Ejemplos de predicciones:")
    mostrarEmpresaConcretaConFilter(df1, DEBUG_FILTRO, "df1")

    mensajeAlerta = ""
    if len(df2a) > 0:
        precision = (100 * len(df1b) / len(df2a))
        if precision >= 25 and len(df2a) >= 10:
            mensajeAlerta = " ==> INTERESANTE"

        print(id_subgrupo + " " + etiqueta + " --> Positivos reales = " + str(
            len(df1a)) + ". Positivos predichos = " + str(len(df2a)) + ". De estos ultimos, los ACIERTOS son: " + str(
            len(df1b)) + " ==> Precision = TP/(TP+FP) = " + str(round(precision, 1)) + " %" + mensajeAlerta)

        print("ENTREGABLEACIERTOSPASADO"
              + "|id_subgrupo:" + str(id_subgrupo)
              + "|escenario:" + etiqueta
              + "|positivosreales:" + str(len(df1a))
              + "|positivospredichos:" + str(len(df2a))
              + "|aciertos:" + str(len(df1b))
              )

        if etiqueta == "TEST" or etiqueta == "VALIDACION":
            dfEmpresasPredichasTrue = pd.merge(dfConIndex, df2a, how="inner", left_index=True, right_index=True)
            dfEmpresasPredichasTrueLoInteresante = dfEmpresasPredichasTrue[
                ["empresa", "antiguedad", "target", "targetpredicho", "anio", "mes", "dia", "hora", "volumen"]]
            # print(tabulate(dfEmpresasPredichasTrueLoInteresante, headers='keys', tablefmt='psql'))

            casosTP = dfEmpresasPredichasTrueLoInteresante[
                dfEmpresasPredichasTrueLoInteresante['target'] == True]  # los BUENOS
            casosFP = dfEmpresasPredichasTrueLoInteresante[
                dfEmpresasPredichasTrueLoInteresante['target'] == False]  # Los MALOS que debemos estudiar y reducir

            # FALSOS POSITIVOS:
            if id_subgrupo != "SG_0":
                print(id_subgrupo + " " + etiqueta +
                      " --> Casos con predicción True y lo real ha sido True (TP, deseados): " + str(casosTP.shape[0]) +
                      " pero tambien hay False (FP, no deseados): " + str(casosFP.shape[0]) + " que son: ")
                falsosPositivosArray = id_subgrupo + "|" + etiqueta + "|" + casosFP.sort_index().index.values
                pathFP = dir_subgrupo + "falsospositivos_" + etiqueta + ".csv"
                print("Guardando lista de falsos positivos en: " + pathFP)
                falsosPositivosArray.tofile(pathFP, sep='\n', format='%s')

                # PREDICCIONES TOTALES (util para el analisis de falsos positivos a posteriori)
                todasLasPrediccionesArray = id_subgrupo + "|" + etiqueta + "|" + dfEmpresasPredichasTrueLoInteresante.sort_index().index.values
                pathTodasPredicciones = dir_subgrupo + "todaslaspredicciones_" + etiqueta + ".csv"
                print(
                    "Guardando lista de todas las predicciones en (verdaderos y falsos positivos) en: " + pathTodasPredicciones)
                todasLasPrediccionesArray.tofile(pathTodasPredicciones, sep='\n', format='%s')

    else:
        print(id_subgrupo + " " + etiqueta + " --> Positivos predichos = " + str(len(df2a)))

    return mensajeAlerta


############ RSI ###################
def relative_strength_idx(df, n=14):
    close = df['close']
    delta = close.diff()
    delta = delta[1:]
    pricesUp = delta.copy()
    pricesDown = delta.copy()
    pricesUp[pricesUp < 0] = 0
    pricesDown[pricesDown > 0] = 0
    rollUp = pricesUp.rolling(n).mean()
    rollDown = pricesDown.abs().rolling(n).mean()
    rs = rollUp / rollDown
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi



##################### DESCARGA DEL SP500 ########################

import requests
from datetime import date
from datetime import datetime as datetime2
from datetime import timedelta

# Etiquetas Globales
FORMATO_FECHA_SP500 = '%Y-%m-%d'
ETIQUETA_FECHA_SP500 = "fecha"
ETIQUETA_CLOSE_SP500 = "close"
ETIQUETA_RENTA_SP500 = "rentaSP500"


def getSP500conRentaTrasXDias(X, fechaInicio, fechaFin, dir_subgrupo):
    # X: días futuros para el cálculo de renta, respecto al día del que se muestran datos (cada fila es distinta)
    # fechaInicio y fechaFin, con formato "2019-01-01"
    # Descarga del histórico del SP500
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&" \
          "graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1168&" \
          "nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=SP500&scale=left&cosd=" + fechaInicio \
          + "&coed=fechaFin&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&" \
            "oet=99999&mma=0&fml=a&fq=Daily%2C%20Close&fam=avg&fgst=lin&fgsnd=2010-04-12&line_index=1&transformation=lin&" \
            "vintage_date=" + fechaFin + "&revision_date=2020-04-10&nd=2010-04-12"
    print("URL: ", url)
    destino = dir_subgrupo + "SP500.csv"
    myfile = requests.get(url, allow_redirects=True)
    open(destino, 'wb').write(myfile.content)

    # Se guarda en dataframe
    datosSP500 = pd.read_csv(filepath_or_buffer=destino, sep=',')

    # Sólo me quedo con las filas cuyo precio sea numérico
    datosSP500 = datosSP500.loc[~(datosSP500['SP500'] == '.')]
    # resetting index
    datosSP500.reset_index(inplace=True)

    # La fecha se convierte a dato de fecha
    dfSP500 = pd.DataFrame(columns=[ETIQUETA_FECHA_SP500, ETIQUETA_CLOSE_SP500, ETIQUETA_RENTA_SP500])
    closeXDiasFuturos = 0
    tamaniodfSP500 = len(datosSP500)
    for index, fila in datosSP500.iterrows():
        fecha = datetime2.strptime(fila['DATE'], FORMATO_FECHA_SP500)
        if index < (tamaniodfSP500 - int(X)):
            filaXDiasFuturos = datosSP500.iloc[index + int(X)]
            closeXDiasFuturos = filaXDiasFuturos['SP500']
            rentaSP500 = 100 * (float(closeXDiasFuturos) - float(fila['SP500'])) / float(closeXDiasFuturos)
        else:
            rentaSP500 = 0

        nuevaFila = [
            {ETIQUETA_FECHA_SP500: fecha, 'close': float(fila['SP500']), ETIQUETA_RENTA_SP500: float(rentaSP500)}]
        dfSP500 = dfSP500.append(nuevaFila, ignore_index=True, sort=False)

    return dfSP500


def anadeComparacionSencillaSP500(sp500, x):
    comparaSP500Ayer=-1000
    fechaAhora = str(x.anio) + "-" + str(x.mes) + "-" + str(x.dia)
    ahora_obj = datetime2.strptime(fechaAhora, FORMATO_FECHA_SP500)
    a = pd.DataFrame()
    # Se itera hacia atrás hasta que se encuentre algún día (laborable) con datos en SP500
    while (a.empty):
        days_to_subtract = 1
        diasAntes_obj = ahora_obj - timedelta(days=days_to_subtract)
        antes = diasAntes_obj.strftime(FORMATO_FECHA_SP500)
        empresaRentaAhora = float(x.close) - float(x.open)
        a=sp500[sp500[ETIQUETA_FECHA_SP500] == antes]
        if(len(a.index)>0):
            sp500RentaAyer = float(a[ETIQUETA_RENTA_SP500])
            comparaSP500Ayer = int(empresaRentaAhora * sp500RentaAyer > 0)
        else:
            ahora_obj=diasAntes_obj

    return comparaSP500Ayer


def descargaTuits(cuentas):
    clavesTwitter = {}
    with open(pathClavesTwitter) as myfile:
        for line in myfile:
            name, var = line.partition("=")[::2]
            clavesTwitter[name.strip()] = var.rstrip()

    # NO DESCOMENTAR, PORQUE SE VERÍAN LAS CLAVES EN EL LOG!!
    # print(clavesTwitter)

    # Twitter API credentials
    api_key = clavesTwitter.get("apikey")
    api_key_secret = clavesTwitter.get("apikeysecret")
    access_token = clavesTwitter.get("accesstoken")
    access_token_secret = clavesTwitter.get("accesstokensecret")

    import os
    import tweepy as tw
    import pandas as pd

    auth = tw.OAuthHandler(api_key, api_key_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth)

    # USO
    api = tweepy.API(auth)

    # Verificación de la conexión
    api.verify_credentials()

    # Se descarga la info de todos los tuiteros
    for cuenta in cuentas:
        get_all_tweets(cuenta, api)


def get_all_tweets(screen_name, api):
    # Twitter only allows access to a users most recent 3240 tweets with this method

    # Se asume que está ya autenticado

    # initialize a list to hold all the tweepy Tweets
    alltweets = []

    # make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name=screen_name, count=200)

    # save most recent tweets
    alltweets.extend(new_tweets)

    # save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    # keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        #print(f"getting tweets before {oldest}")

        # all subsequent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name=screen_name, count=200, max_id=oldest)

        # save most recent tweets
        alltweets.extend(new_tweets)

        # update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        #print(f"...{len(alltweets)} tweets downloaded so far")

    # transform the tweepy tweets into a 2D array that will populate the csv
    outtweets = [[tweet.id_str, tweet.created_at, tweet.text] for tweet in alltweets]

    # write the csv
    with open(pathDestinoTweets + "/" + f'{screen_name}' + "_tweets.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "created_at", "text"])
        writer.writerows(outtweets)

    pass


def vaciarCarpeta(folder):
    import os, shutil
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def anadeMencionesTwitterPorTuiteros(diasAntiguedad, tuits, x, stringABuscar):
    numeroMencionesTotales = 0
    fechaRowActual = str(x.anio) + "-" + str(x.mes) + "-" + str(x.dia)
    fechaRowActual_obj= datetime2.strptime(fechaRowActual, FORMATO_FECHA_TWITTER)
    fechaElegida_obj = fechaRowActual_obj - timedelta(days=diasAntiguedad)
    fechaElegida = fechaElegida_obj.strftime(FORMATO_FECHA_TWITTER)

    # Para optimizar recursos, se eliminan todos los tuits que no coincidan con la fecha indicada
    tuits=tuits.loc[tuits["created_at"].str.startswith(fechaElegida, na=False)]

    ficheros = os.listdir(pathDestinoTweets)
    tuiteros=tuits.tuitero.unique()

    for tuitero in tuiteros:
        tuitsDeTuitero=tuits[tuits['tuitero'] == tuitero]
        numeroMencionesPorElTuitero=0

        tuitsDeTuitero = tuitsDeTuitero.loc[tuitsDeTuitero["text"].str.contains(stringABuscar)]

        # Se comprueba si hay al menos 1 tuit
        if tuitsDeTuitero.shape[0] > 0:
            numeroMencionesTotales += 1
            # print("El tuitero "+ tuitero + "ha escrito al menos un tuit de "+stringABuscar+
            #       " en la fecha " + fechaElegida)

    # print("Número de tuiteros que referencian al menos una vez a la empresa " + x.empresa + " en la fecha "
    #       + fechaElegida+ " : "+numeroMencionesTotales)
    return numeroMencionesTotales


def anadeFeatureTwitter(entradaFeaturesYTarget, tituloNuevaFeature, pathDestinoTweets, cuentas,
                                        antiguedadMaxima, stringOpcionalEnTuit):

    # Se vacía la carpeta donde se guardarán los tuits, y se descargan los tuits de tuiteros
    vaciarCarpeta(pathDestinoTweets)
    descargaTuits(cuentas)

    # Se acumulan los tuits en memoria, para optimizar velocidad
    ficheros = os.listdir(pathDestinoTweets)
    tuits = pd.DataFrame(columns=['id', 'created_at', 'text', 'tuitero'])
    for fichero in ficheros:
        pathFichero = os.path.join(pathDestinoTweets, fichero)
        if os.path.isfile(pathFichero) and fichero.endswith('.csv'):
            numeroMencionesPorElTuitero = 0
            # Cada fichero pertenece a un tuitero
            tuitero = fichero.split("_")[0]
            # Se lee el fichero
            tuitsDeFichero = pd.read_csv(filepath_or_buffer=pathFichero, sep=',')
            # Para acelerar el proceso, se toman sólo las filas que contengan "$" en su columna text (mensaje),
            # ya que siempre buscamos un ticker.
            tuitsDeFichero=tuitsDeFichero[tuitsDeFichero['text'].str.contains("\$")]

            # Se permite filtrar opcionalmente por un string, si no está vacío.
            if not stringOpcionalEnTuit:
                tuitsDeFichero = tuitsDeFichero[tuitsDeFichero['text'].str.contains(stringOpcionalEnTuit)]

            # Se añade el nombre del tuitero en la primera columna de todas las filas del dataframe
            tuitsDeFichero['tuitero'] = pd.Series([tuitero for x in range(len(tuitsDeFichero.index))])
            # Es una chapuza, pero son pocos ficheros
            tuits = tuits.append(tuitsDeFichero, ignore_index=True)

        # Se sumarán las apariciones en todos los días del rango desde 0 a antiguedadMaxima. Por defecto serán 0
        entradaFeaturesYTarget[tituloNuevaFeature] = 0
        for i in range(0, antiguedadMaxima):
            # print("Para crear la nueva feature "+tituloNuevaFeature+
            #       " se añaden menciones en TWITTER de la empresa en el fichero "+fichero+
            #       " con antigüedad "+str(i)+"...")
            # Se busca una mención en el tuit a la empresa de cada fila manejada por nuestro dataframe
            entradaFeaturesYTarget[tituloNuevaFeature] += entradaFeaturesYTarget.apply(
                lambda x: anadeMencionesTwitterPorTuiteros(i, tuits, x, "\$" + x.empresa), axis=1)

    return entradaFeaturesYTarget




def aniadirColumnasDependientesSP500 ():
    x=0
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

def aniadirColumnasDeTwitter():
    x=0

    # SE QUITAN PORQUE PARA 100 EMPRESAS TARDA 10 MINUTOS POR SUBGRUPO, Y APENAS INFLUYE
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

'''
Crea una imagen de las función de densidad de probabilidad de cada columna (feature) del dataframe.
'''
def pintarFuncionesDeDensidad (miDF, dir_subgrupo_img, dibujoBins, descripcion):
    print("FUNCIONES DE DENSIDAD (" + descripcion + "):")
    for column in miDF:
        path_dibujo = dir_subgrupo_img + column + ".png"
        print("Guardando distrib de col: " + column + " en fichero: " + path_dibujo)
        datos_columna = miDF[column]
        sns.distplot(datos_columna, kde=False, color='red', bins=dibujoBins)
        plt.title(column, fontsize=10)
        plt.savefig(path_dibujo, bbox_inches='tight')
        plt.clf();
        plt.cla();
        plt.close()  # Limpiando dibujo




