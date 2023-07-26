import glob
import os
import shutil
import sys
import pandas as pd
import pandas_profiling.model.describe
from tabulate import tabulate
from pandas_profiling import ProfileReport

#  Se usan los PERIODOS típicos que suelen usar los robots (consideraremos velas)
periodosDParaParametros = [4, 7, 20, 50]

########### FUNCIONES #######################################################################################
"""
Calcula la columna especial TARGET y la añade al dataframe de entrada.
Analiza los periodos [t1,t2] y [t2,t3], donde el foco está en el tiempo t1. El precio debe subir durante [t1,t2] y no haber bajado demasiado en [t2,t3].
	 
Parametros:
    - entradaDF: dataframe de entrada. Debe contener la columna de precio de cierre (close)
    - nombreEmpresa: nombre de la empresa analizada
    - S: Subida del precio de cierre durante [t1,t2] en el HIGH (no en el close)
    - X: Duración del periodo [t1,t2]
    - R (NO USADO): Caida ligera máxima permitida durante [t2,t3], en TODAS esas velas
    - M: Duración del periodo [t2,t3]. garantiza estabilidad de la subida en [t1, t2]
    - F: Caida ligera permitida durante [t2,t3], en la ÚLTIMA vela en su CLOSE
    - B (NO USADO): Caida ligera permitida durante [t1,t2], en TODAS esas velas
    - umbralMaximo: Porcentaje máximo aceptado para la subida de cada vela respecto del incremento medio en las velas de 1 a X. Sirve para QUEDARSE solo con PELOTAZOS. Recomendable: 0 (3 o menos).
    - umbralMinimo: Porcentaje mínimo aceptado para la subida de cada vela respecto del incremento medio en las velas de 1 a X. Sirve para DESCARTAR subidas enormes no controladas. Recomendable: 5 (3 o mas).
"""
def calcularTargetYanhadirlo (entradaDF, nombreEmpresa, S, X, R, M, F, B, umbralMaximo, umbralMinimo):
    #print("Calculando columna TARGET...")
    tempDF = pd.DataFrame()  # auxiliar
    tempDF['antiguedad'] = entradaDF['antiguedad']
    tempDF['anio'] = entradaDF['anio']
    tempDF['mes'] = entradaDF['mes']
    tempDF['dia'] = entradaDF['dia']

    tempDF['close'] = entradaDF['close']                # Precio de cierre del día analizado t1
    tempDF['high'] = entradaDF['high']                # Precio de cierre del día analizado t1
    tempDF['high_max_durante_X'] = entradaDF['high'].copy().rolling(X+1, min_periods=X+1).max()  # Maximo precio high en ventana [t1,t2]
    tempDF['close_X'] = entradaDF['close'].copy().shift(X)     # Precio de cierre del día analizado t2 (t2=t1+X)
    #tempDF['high_en_X'] = entradaDF['high'].copy().shift(X)  # Precio de cierre del día analizado t2 (t2=t1+X)
    #tempDF['high_durante_XM'] = entradaDF['high'].copy().rolling(X+M+1,min_periods=(X+M+1)).max()  # Maximo precio high en ventana [t1,t3]
    tempDF['close_XM'] = entradaDF['close'].copy().shift(X + M)  # Precio de cierre del día analizado t3 (t3=t1+X+M)
    #tempDF['high_XM'] = entradaDF['high'].copy().shift(X + M)  # Precio de cierre del día analizado t3 (t3=t1+X+M)


    condicion1 = pd.DataFrame(tempDF['high_max_durante_X'] > ((100 + S) / 100 * tempDF['close']) )  # corte del umbral S en algun HIGH de alguna vela [t1,t2]
    condicion2 = pd.DataFrame(tempDF['close_X'] > ((100 + S - F) / 100 * tempDF['close']))  # corte del umbral S en algun CLOSE en t1
    condicion3 = pd.DataFrame(tempDF['close_XM'] > ((100 + S - F) / 100 * tempDF['close']) )  # corte del umbral S en algun CLOSE en t3
    tempDF['TARGET'] = condicion1 & condicion2 & condicion3

    tempDF.replace({False: 0, True: 1}, inplace=True)
    numPositivos = len(tempDF[tempDF['TARGET'] == True])
    numNegativos = len(tempDF[tempDF['TARGET'] == False])
    tasaDesbalanceo = round(numPositivos/numNegativos, 2)
    print("ElaboradosUtils.py - Empresa: " + nombreEmpresa + " \t-> Tasa de desbalanceo (true/false): \t" + str(numPositivos) + "/" + str(numNegativos) + " \t= " + str(tasaDesbalanceo))

    #Se añade la columna al final del dataframe de entrada
    entradaDF['TARGET'] = tempDF['TARGET']


"""
Lee CSV limpio y escribe un CSV nuevo con columnas elaboradas generadas aqui.
Un columna elaborada especial es el TARGET.
"""
def procesarCSV (pathEntrada, pathSalida, modoTiempo, analizarEntrada, S, X, R, M, F, B, umbralMaximo, umbralMinimo):
    # print("Construyendo elaborados: " + pathEntrada + " --> " + pathSalida)
    entradaDF = pd.read_csv(filepath_or_buffer=pathEntrada, sep='|')
    nombreEmpresa = pathEntrada.split("/")[-1].replace(".csv", "")
    tempDF = pd.DataFrame()  # auxiliar

    # Analisis breve de la entrada
    if analizarEntrada:
        print("****** analizarEntrada: " + pathEntrada + " ********")
        print(tabulate(entradaDF.head(n=3), headers='keys', tablefmt='psql'))
        print("PANDAS-DESCRIBE: ")
        print(tabulate(entradaDF.describe(), headers='keys', tablefmt='psql'))
        print("PANDAS-PROFILING: ")
        muestraEntradaDF = entradaDF
        prof = ProfileReport(entradaDF)
        prof.to_file(output_file="/bolsa/logs/"+modoTiempo+"_"+nombreEmpresa+".html")


    # GAP: compara cierre con la apertura al día siguiente (periodo = 1 vela)
    tempDF['open'] = entradaDF['open']  # apertura al dia analizado
    tempDF['close_dia_anterior'] = entradaDF['close'].shift(-1)
    entradaDF['gap_apertura'] = 100 * (tempDF['open'] - tempDF['close_dia_anterior']) / tempDF['close_dia_anterior']

    # SPAN high-low respecto al precio CLOSE, en la vela analizada
    entradaDF['span_diario'] = 100 * (entradaDF['high'] - entradaDF['low']) / entradaDF['close']

    # Nuevas columnas calculadas para cada periodo P
    for periodo in periodosDParaParametros:

        # PENDIENTE RELATIVA DE CRECIMIENTO de PRECIO CLOSE en ultimas PERIODO velas (respecto del precio close inicial)
        tempDF['close'] = entradaDF['close']
        tempDF['close_shifted_'+str(periodo)] = entradaDF['close'].shift(-1 * periodo)
        entradaDF['close_pendienterelativa_' + str(periodo)] = 100 * (tempDF['close'] - tempDF['close_shifted_'+str(periodo)] )/ tempDF['close_shifted_'+str(periodo)]

        # PENDIENTE RELATIVA DE CRECIMIENTO de VOLUMEN en ultimas PERIODO velas (respecto del volumen habitual, medio historico de la empresa)
        tempDF['volumen'] = entradaDF['volumen']
        volumen_medio_historico = tempDF['volumen'].mean()
        tempDF['volumen_shifted_' + str(periodo)] = entradaDF['volumen'].shift(-1 * periodo)
        entradaDF['VARREL_'+str(periodo)+'_VOLUMEN'] = 100 * (tempDF['volumen'] - tempDF['volumen_shifted_' + str(periodo)]) / volumen_medio_historico

        # SPAN entre precios HIGH y LOW en últimas PERIODO velas (respecto del precio close inicial)
        # Para la vela analizada en tiempo t, se leen todos los precios high del periodo [t-P, t] cogiendo el máximo; y análogo para coger el mínimo de los precios low.
        # Después, el span es la resta: span = max(high) - min(low)
        # Y se calcula relativo al precio close de la vela analizada en tiempo t
        tempDF['high'] = entradaDF['high']
        entradaDF['spanhigh_durante_periodo'] = 100 * (tempDF['high'].copy().rolling(periodo+1).max().shift(-1*periodo) - tempDF['close'] )/ tempDF['close']  # Maximo precio high en el periodo (ROLLING BACKWARDS)
        tempDF['low'] = entradaDF['low']
        entradaDF['spanlow_durante_periodo'] = 100 * (tempDF['low'].copy().rolling(periodo+1).min().shift(-1*periodo) - tempDF['close'] )/ tempDF['close']  # Minimo precio low en el periodo  (ROLLING BACKWARDS)

        # Variación de MEDIA EWMA (exponentially weighted moving average) sobre precios HIGH, relativa al precio CLOSE de la vela analizada: si los picos HIGH se disparan respecto al cierre, pronto subirá el close
        entradaDF['ewma_highclose_' + str(periodo)] = 100 * (entradaDF['high'].ewm(span=periodo, adjust=False).mean() - entradaDF['close'] )/ entradaDF['close']
        # Variación de MEDIA EWMA (exponentially weighted moving average) sobre precios LOW, relativa al precio CLOSE de la vela analizada: si los picos LOW se disparan respecto al cierre, pronto caera el close
        entradaDF['ewma_lowclose_' + str(periodo)] = 100 * (entradaDF['low'].ewm(span=periodo, adjust=False).mean() - entradaDF['close']) / entradaDF['close']

        # Variación de MEDIA SIMPLE (SMA) sobre precios HIGH, relativa al precio CLOSE de la vela analizada: si los picos HIGH se disparan respecto al cierre, pronto subirá el close
        entradaDF['sma_highclose_' + str(periodo)] = 100 * (
                    entradaDF['high'].rolling(periodo+1).mean().shift(-1*periodo) - entradaDF['close']) / entradaDF['close']
        # Variación de MEDIA SIMPLE (SMA) sobre precios LOW, relativa al precio CLOSE de la vela analizada: si los picos LOW se disparan respecto al cierre, pronto caera el close
        entradaDF['sma_lowclose_' + str(periodo)] = 100 * (
                    entradaDF['low'].rolling(periodo+1).mean().shift(-1*periodo) - entradaDF['close']) / entradaDF['close']

    ##################  COLUMNA ESPECIAL: TARGET #######################
    if "pasado" == modoTiempo:
        calcularTargetYanhadirlo(entradaDF, nombreEmpresa, S, X, R, M, F, B, umbralMaximo, umbralMinimo)
    else: #futuro
        entradaDF['TARGET'] = 'null'


    # Pintar estado final del dataframe CON columnas ELABORADAS:
    #print(tabulate(entradaDF.head(n=10), headers='keys', tablefmt='psql'))

    #Guardar
    # print("Justo antes de guardar, entradaDF: " + str(entradaDF.shape[0]) + " x " + str(entradaDF.shape[1]) + " --> " + pathSalida)
    entradaDF.to_csv(pathSalida, index=False, sep='|', float_format='%.4f')

