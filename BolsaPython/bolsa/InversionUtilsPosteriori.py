import sys
import os
import pandas as pd
import glob
import numpy as np

#EXPLICACIÓN:
#1º. inversion.sh ha generado unos 100 ficheros al dia: 50 son de 300KB (tienen un pequeño histórico) y otros son de 5KB (antiguedad=0)
#2º. Entonces, en PYTHON, vamos a recorrer todos los CSV de 5KB y vamos a construir una tabla (DATAFRAME) que tenga -->  DIA_INVERSION, DIA_POSTERIORI, subgrupo, empresa, target_predicho, target_predicho_PROB
#3º. En PYTHON, vamos a recorrer todos los CSVs de 300 KB y vamos a construir una tabla (DATAFRAME) que tenga --> DIA, EMPRESA, precio_close, target (es el target REAL)
#4º. Comparamos las dos tablas --> En la tabla 1 metemos la columna de qué ha pasado realmente en el futuro (precio, target REAL)
#5º. Graficas del rendimiento...

print("--- InversionUtilsPosteriori: INICIO ---")

dirDropbox = sys.argv[1]
dirAnalisis = sys.argv[2]
S = sys.argv[3]
X = sys.argv[4]
R = sys.argv[5]
M = sys.argv[6]
F = sys.argv[7]
B = sys.argv[8]

# Se listan los ficheros manejables y grandes (sólo los má recientes, con antigüedad 0 en el nombre)
ficherosManejables = glob.glob(dirDropbox+"*MANEJABLE*")
ficherosGrandesCero = glob.glob(dirDropbox+"*GRANDE_0*")

#Se iteran los ficheros manejables, para leer su contenido
columnasManejables=['empresa', 'antiguedad', 'anio', 'mes', 'dia', 'close', 'TARGET_PREDICHO', 'TARGET_PREDICHO_PROB']
datosManejables=pd.DataFrame()
contenidosManejables=[]

for filename in ficherosManejables:
    datosFichero = pd.read_csv(filepath_or_buffer=filename, sep='|')
    datosFicheroReducidos=datosFichero[columnasManejables]
    datosFicheroInteresantes=pd.DataFrame(datosFicheroReducidos.loc[datosFicheroReducidos['TARGET_PREDICHO'] == 1],
                                          columns=columnasManejables)
    #Debo añadir una columna con el número de subgrupo
    datosFicheroInteresantes['subgrupo']=filename[(filename.rfind('SG')+3):(filename.rfind('SG')+5)].replace('_', '')
    datosManejables=datosManejables.append(datosFicheroInteresantes)



#Se iteran los ficheros grandes, para extraer su contenido
columnasGrandes=['empresa', 'antiguedad', 'anio', 'mes', 'dia', 'close', 'TARGET']
datosGrandes=pd.DataFrame()
contenidosGrandes=[]
ficherosGrandesCeroAAnalizar=[]
fechaTemporal=0
#Se leerán sólo los de la fecha más reciente de entre todos los ficheros grandes
for filename in ficherosGrandesCero:
    fechaFichero=int(filename[(filename.rfind('/')+1):(filename.rfind('/')+9)])
    #Se tomará sólo el fichero grande más reciente
    if(fechaTemporal<fechaFichero):
        ficherosGrandesCeroAAnalizar.append(filename)

for filename in ficherosGrandesCeroAAnalizar:
    datosFichero = pd.read_csv(filepath_or_buffer=filename, sep='|')
    datosFicheroReducidos=datosFichero[columnasGrandes]
    datosFicheroInteresantes=pd.DataFrame(datosFicheroReducidos, columns=columnasGrandes)
    datosFicheroInteresantes['subgrupo'] = filename[(filename.rfind('SG') + 3):(filename.rfind('SG') + 5)].replace('_',
                                                                                                                   '')
    datosGrandes=datosGrandes.append(datosFicheroInteresantes)

#OBTENCIÓN DEL CLOSE PARA LAS FILAS PREDICHAS TRAS X DÍAS POSTERIORES
datosMergeados = pd.merge(datosGrandes, datosManejables, how='right', on=['empresa', 'anio', 'mes', 'dia', 'close', 'subgrupo'])
datosDesplazados=datosMergeados
datosDesplazados.loc[:, 'antiguedad_x'] -= int(X)
datosDesplazados['antiguedad'] = datosDesplazados['antiguedad_x']
datosFuturo=pd.merge(datosGrandes, datosDesplazados, how='right', on=['empresa', 'antiguedad', 'subgrupo'])
datosAAnalizar=datosFuturo.loc[datosFuturo['TARGET_y'].isin(['1', '0'])] # Son datos tan antiguos que sí tienen su resultado futuro (que es el REAL)
datosAAnalizar.loc[:, 'antiguedad_x'] += int(X)

#CÁLCULO DE RENDIMIENTO MEDIO POR FECHA Y SUBGRUPO
#En fecha_x está el futuro. En fecha_y está el dato predicho. Se genera una columna nueva que obtiene el rendimiento real (close_x vs close_y)
datosAAnalizar.loc[:, 'rendimiento'] = 100 * (datosAAnalizar['close_x']-datosAAnalizar['close_y'])/datosAAnalizar['close_y']

grupos=datosAAnalizar.groupby(['mes_y', 'dia_y', 'subgrupo'])
#Se calculan probabilidades, y se loggean/grafican por antiguedad
for group_name, df_group in grupos:
    #En cada antigüedad se reinician los contadores
    rentaMedia=0
    rentasAcumuladas=[]
    anio=0
    mes=0
    dia=0

    for row_index, row in df_group.iterrows():
        rentasAcumuladas.append(row['rendimiento'])
        anio=row['anio_y']
        mes = row['mes_y']
        dia = row['dia_y']
        subgrupo = row['subgrupo']

    # Se calculan las rentas
    rentaMedia = np.mean(rentasAcumuladas)

    #Se imprimen los resultados
    print('\nANIO/MES/DIA: {:.0f}'.format(anio)+"/"+'{:.0f}'.format(mes)+"/"+'{:.0f}'.format(dia))
    print(' SUBGRUPO {}'.format(subgrupo))
    print(' RENTABILIDAD MEDIA {}'.format(rentaMedia))

print("\n--- InversionUtilsPosteriori: FIN ---")



