import sys
import os
import pandas as pd
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates

#EXPLICACIÓN:
#Se recorren los ficheros grandes y manejables, y se saca el rendimiento medio, por día y subgrupo.
# Se vuelcan en log, excel y dibujo

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

resultadoAnalisis=pd.DataFrame(columns=["fecha", "anio", "mes", "dia", "subgrupo", "precisionMedia", "rentaMedia"])

#Se calculan probabilidades, y se loggean/grafican por antiguedad
for group_name, df_group in grupos:
    #En cada antigüedad se reinician los contadores
    rentaMedia=0
    rentasAcumuladas=[]
    anio=0
    mes=0
    dia=0
    numPrecisionAcertada=0
    numPrecisionFallada=0
    precisionMedia=-1

    for row_index, row in df_group.iterrows():
        rentasAcumuladas.append(row['rendimiento'])
        anio=row['anio_y']
        mes = row['mes_y']
        dia = row['dia_y']
        subgrupo = row['subgrupo']
        target = row['TARGET_y']
        if target == 1:
            numPrecisionAcertada += 1
        elif target == 0:
            numPrecisionFallada += 1

    # Se calcula la precisión
    if (numPrecisionAcertada + numPrecisionFallada) > 0:
        precisionMedia = numPrecisionAcertada / (numPrecisionAcertada + numPrecisionFallada)

    # Se calcula la renta media
    rentaMedia = np.mean(rentasAcumuladas)

    # Se imprimen los resultados
    print('\nANIO/MES/DIA: {:.0f}'.format(anio)+"/"+'{:.0f}'.format(mes)+"/"+'{:.0f}'.format(dia))
    print(' SUBGRUPO: {}'.format(subgrupo))
    print(' PRECISION MEDIA: {:.0f}'.format(numPrecisionAcertada)+'/'+'{:.0f}'.format(numPrecisionAcertada+numPrecisionFallada)+' = {:.2f}'.format(precisionMedia))
    print(' RENTABILIDAD MEDIA: {:.2f}'.format(rentaMedia))

    # Se guardan los resultados en un DataFrame
    datetime_str = str(dia)+'/'+str(mes)+'/'+str(anio)
    fecha = datetime.strptime(datetime_str, '%d/%m/%Y')

    nuevaFila = [{'fecha': fecha, 'anio': anio, 'mes': mes, 'dia': dia, 'subgrupo': subgrupo, 'precisionMedia': precisionMedia, 'rentaMedia': rentaMedia}]
    resultadoAnalisis=resultadoAnalisis.append(nuevaFila,ignore_index=True,sort=False)

# Se escriben los resultados a un Excel
pathCsvResultados=dirAnalisis+"RESULTADOS_ANALISIS.csv"
print("Guardando: " + pathCsvResultados)
resultadoAnalisis.to_csv(pathCsvResultados, index=False, sep='|')

# Se pintan en dos gráficas: precisión media y rentabilidad media
resultadoAnalisis['aniomesdia']=10000*resultadoAnalisis['anio']+100*resultadoAnalisis['mes']+resultadoAnalisis['dia']
subgrupos=resultadoAnalisis['subgrupo'].unique().tolist()

# PRECISIÓN MEDIA
ax = plt.gca()  # gca significa 'get current axis'
for subgrupo in subgrupos:
    resultadoPorSubgrupo=resultadoAnalisis.loc[resultadoAnalisis['subgrupo'] == subgrupo]
    resultadoPorSubgrupo.plot(kind='line', x='fecha', y='precisionMedia', ax=ax, label=subgrupo, marker="+")
ax.tick_params(axis='x', labelrotation=20)  # Rota las etiquetas del eje X
formatter = mdates.DateFormatter("%Y-%m-%d")
ax.xaxis.set_major_formatter(formatter)
locator = mdates.DayLocator()
ax.xaxis.set_major_locator(locator)
plt.title('Evolución de la PRECISION cada modelo entrenado de subgrupo')
plt.xticks(rotation=90, ha='right')
#plt.show()
plt.savefig(dirAnalisis+"precision.png")
plt.close()

# RENTABILIDAD MEDIA
ax2 = plt.gca()  # gca significa 'get current axis'
for subgrupo in subgrupos:
    resultadoPorSubgrupo=resultadoAnalisis.loc[resultadoAnalisis['subgrupo'] == subgrupo]
    resultadoPorSubgrupo.plot(kind='line', x='fecha', y='rentaMedia', ax=ax2, label=subgrupo, marker="+")
ax2.tick_params(axis='x', labelrotation=20)  # Rota las etiquetas del eje X
formatter = mdates.DateFormatter("%Y-%m-%d")
ax2.xaxis.set_major_formatter(formatter)
locator = mdates.DayLocator()
ax2.xaxis.set_major_locator(locator)
plt.title('Evolución de la RENTABILIDAD cada modelo entrenado de subgrupo')
plt.xticks(rotation=90, ha='right')
#plt.show()
plt.savefig(dirAnalisis+"rentabilidad.png")
plt.close()

print("\n--- InversionUtilsPosteriori: FIN ---")



