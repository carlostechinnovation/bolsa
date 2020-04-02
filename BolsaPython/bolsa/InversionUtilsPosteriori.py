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
columnasManejables=['empresa', 'antiguedad', 'anio', 'mes', 'dia', 'hora', 'close', 'TARGET_PREDICHO', 'TARGET_PREDICHO_PROB']
datosManejables=pd.DataFrame()
contenidosManejables=[]

for filename in ficherosManejables:
    datosFichero = pd.read_csv(filepath_or_buffer=filename, sep='|')
    datosFicheroReducidos=datosFichero[columnasManejables]
    datosFicheroInteresantes=pd.DataFrame(datosFicheroReducidos.loc[datosFicheroReducidos['TARGET_PREDICHO'] == 1],
                                          columns=columnasManejables)
    contenidosManejables.append(datosFicheroInteresantes)

datosManejables=datosManejables.append(contenidosManejables)

#Se iteran los ficheros grandes, para extraer su contenido
columnasGrandes=['empresa', 'antiguedad', 'anio', 'mes', 'dia', 'hora', 'close', 'TARGET']
datosGrandes=pd.DataFrame()
contenidosGrandes=[]

for filename in ficherosGrandesCero:
    datosFichero = pd.read_csv(filepath_or_buffer=filename, sep='|')
    datosFicheroReducidos=datosFichero[columnasGrandes]
    datosFicheroInteresantes=pd.DataFrame(datosFicheroReducidos, columns=columnasGrandes)
    contenidosGrandes.append(datosFicheroInteresantes)

datosGrandes=datosGrandes.append(contenidosGrandes)

#Se mergean los datos, separados por dia y subgrupo
#Se ubican los manejables dentro de los grandes. Así puedo conocer su antigüedad
datosMergeados = pd.merge(datosGrandes, datosManejables, how='right', on=['empresa', 'anio', 'mes', 'dia', 'hora', 'close'])

#De entre los grandes, se escogen aquéllos cuya antigüedad sea la de los "manejables - X" (vamos hacia el futuro)
datosPredichos=pd.merge(datosGrandes, datosManejables, how='left', on=['empresa'])
#Desplazo la antiguedad de los mergeados: resto X velas
datosMergeados['antiguedad_x'] -= int(X)
# Relleno, de entre los grandes predichos, los de empresa y antiguedad desplazada X
datosSeleccionados=pd.merge(datosPredichos, datosMergeados, how='left', on=['empresa', 'antiguedad_x'])

#Se filtran los que tengan target real (calculado con la realidad futura) rellena
datosAAnalizar=datosSeleccionados.loc[datosSeleccionados['TARGET_y'] == 1]

#Se clasifican/agrupan los datos por mes+día (PENDIENTE HACERLO TAMBIÉN POR SUBGRUPO)
#En fecha_x está el futuro. En fecha_y está el dato predicho. Se genera una columna nueva que obtiene el rendimiento real (close_x vs close_y)
datosAAnalizar['rendimiento'] = 100 * (datosAAnalizar['close_x']-datosAAnalizar['close_y'])/datosAAnalizar['close_y']

grupos=datosAAnalizar.groupby('antiguedad_x')
#Se calculan probabilidades, y se loggean/grafican por antiguedad
for group_name, df_group in grupos:
    #En cada antigüedad se reinician los contadores
    rentaMedia=0
    rentasAcumuladas=[]

    for row_index, row in df_group.iterrows():
        rentasAcumuladas.append(row['rendimiento'])

    # Se calculan las rentas
    rentaMedia = np.mean(rentasAcumuladas)

    #Se imprimen los resultados
    print('\nANTIGUEDAD {}: '.format(group_name))
    print(' RENTABILIDAD MEDIA {}'.format(rentaMedia))

print("\n--- InversionUtilsPosteriori: FIN ---")



