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
    - S: Subida del precio de cierre durante [t1,t2]
    - X: Duración del periodo [t1,t2]
    - R: Caida ligera máxima permitida durante [t2,t3], en TODAS esas velas.
    - M: Duración del periodo [t2,t3]
    - F: Caida ligera permitida durante [t2,t3], en la ÚLTIMA vela.
    - B: Caida ligera permitida durante [t1,t2], en TODAS esas velas.
    - umbralMaximo: Porcentaje máximo aceptado para la subida de cada vela respecto del incremento medio en las velas de 1 a X.
    - umbralMinimo: Porcentaje mínimo aceptado para la subida de cada vela respecto del incremento medio en las velas de 1 a X.
"""
def calcularTargetYanhadirlo (entradaDF, nombreEmpresa, S, X, R, M, F, B, umbralMaximo, umbralMinimo):
    print("Calculando columna TARGET...")



def procesarCSV (pathEntrada, pathSalida, modoTiempo, analizarEntrada, S, X, R, M, F, B, umbralMaximo, umbralMinimo):
    print("Construyendo elaborados: " + pathEntrada + " --> " + pathSalida)
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

        # PENDIENTE RELATIVA DE CRECIMIENTO de PRECIO CLOSE en ultimas PERIODO velas
        tempDF['close'] = entradaDF['close']
        tempDF['close_shifted_'+str(periodo)] = entradaDF['close'].shift(-1 * periodo)
        entradaDF['close_pendienterelativa_' + str(periodo)] = 100 * (tempDF['close'] - tempDF['close_shifted_'+str(periodo)] )/ tempDF['close_shifted_'+str(periodo)]

        # PENDIENTE RELATIVA DE CRECIMIENTO de VOLUMEN en ultimas PERIODO velas
        tempDF['volumen'] = entradaDF['volumen']
        tempDF['volumen_shifted_' + str(periodo)] = entradaDF['volumen'].shift(-1 * periodo)
        entradaDF['volumen_pendienterelativa_' + str(periodo)] = 100 * (tempDF['volumen'] - tempDF['volumen_shifted_' + str(periodo)]) / tempDF['volumen_shifted_' + str(periodo)]

        # SPAN entre precios High y Low en últimas PERIODO velas.
        # Para la vela analizada en tiempo t, se leen todos los precios high del periodo [t-P, t] cogiendo el máximo; y análogo para coger el mínimo de los precios low.
        # Después, el span es la resta: span = max(high) - min(low)
        # Y se calcula relativo al precio close de la vela analizada en tiempo t
        # tempDF['high'] = entradaDF['high']
        # tempDF['volumen_shifted_' + str(periodo)] = entradaDF['volumen'].shift(-1 * periodo)
        # entradaDF['volumen_pendienterelativa_' + str(periodo)] = (tempDF['volumen'] - tempDF[
        #     'volumen_shifted_' + str(periodo)]) / tempDF['volumen_shifted_' + str(periodo)]

        # Variación de MEDIA EWMA (exponentially weighted moving average) sobre precios CLOSE, relativa al precio close de la vela analizada
        entradaDF['close_ewma_' + str(periodo)] = 100 * (entradaDF['close'].ewm(span=periodo, adjust=False).mean() - entradaDF['close'] )/ entradaDF['close']



    ##################  COLUMNA ESPECIAL: TARGET #######################
    if "pasado" == modoTiempo:
        calcularTargetYanhadirlo(entradaDF, nombreEmpresa, S, X, R, M, F, B, umbralMaximo, umbralMinimo)


    # Pintar estado final del dataframe CON columnas ELABORADAS:
    print(tabulate(entradaDF.head(n=10), headers='keys', tablefmt='psql'))