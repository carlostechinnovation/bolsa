import sys

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import requests
from datetime import date
import math

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

#------------------------Parámetros análisis---------------
probUnoMinima=0.8

#----------------------FUNCIONES-----------------------------
def pintar(resultadoAnalisis, subgrupos, Y):
    # Se pintan en dos gráficas: precisión media y rentabilidad media

    # PRECISIÓN MEDIA
    ax = plt.gca()  # gca significa 'get current axis'
    for subgrupo in subgrupos:
        resultadoPorSubgrupo=resultadoAnalisis.loc[resultadoAnalisis['subgrupo'] == subgrupo]
        # Se eliminan las precisiones < 0, que son no válidas
        precisiones=resultadoPorSubgrupo['precisionMedia']
        precisiones = precisiones[precisiones > 0]
        media = np.mean(precisiones)
        numeroDiasAnalizados=len(precisiones)
        # El plot no me deja no pintar los inválidos: valor -1
        resultadoPorSubgrupo.plot(kind='line', x='fecha', y='precisionMedia', ax=ax, label=subgrupo+" --> "+'{:.0f}%'.format(100*media)+', {:.0f}'.format(numeroDiasAnalizados)+' compras', marker="+")
    ax.tick_params(axis='x', labelrotation=20)  # Rota las etiquetas del eje X
    formatter = mdates.DateFormatter("%Y-%m-%d")
    #ax.xaxis.set_major_formatter(formatter)
    locator = mdates.DayLocator()
    #ax.xaxis.set_major_locator(locator)
    plt.title('PRECISION por subgrupo y fecha', fontsize=10)
    plt.xticks(rotation=90, ha='right')
    #plt.show()
    plt.savefig(dirAnalisis +"precision-esperando " + str(Y) + " días.png")
    plt.close()

    # RENTABILIDAD MEDIA
    ax2 = plt.gca()  # gca significa 'get current axis'
    for subgrupo in subgrupos:
        resultadoPorSubgrupo=resultadoAnalisis.loc[resultadoAnalisis['subgrupo'] == subgrupo]
        resultadoPorSubgrupo['rentaMedia'] = resultadoPorSubgrupo['rentaMedia'] / int(Y)
        media = np.mean(resultadoPorSubgrupo['rentaMedia'])
        numeroDiasAnalizados=len(resultadoPorSubgrupo['rentaMedia'])
        resultadoPorSubgrupo.plot(kind='line', x='fecha', y='rentaMedia', ax=ax2, label=subgrupo+" --> "+'{:.1f}%'.format(media)+', {:.0f}'.format(numeroDiasAnalizados)+' compras', marker="+")
    ax2.tick_params(axis='x', labelrotation=20)  # Rota las etiquetas del eje X
    formatter = mdates.DateFormatter("%Y-%m-%d")
    #ax2.xaxis.set_major_formatter(formatter)
    locator = mdates.DayLocator()
    #ax2.xaxis.set_major_locator(locator)
    plt.title('RENTA DIARIA por subgrupo y fecha', fontsize=10)
    plt.xticks(rotation=90, ha='right')
    #plt.show()
    plt.savefig(dirAnalisis +"rentabilidad-esperando " + str(Y) + " días.png")
    plt.close()

    # RENTABILIDAD ACUMULADA
    ax3 = plt.gca()  # gca significa 'get current axis'
    for subgrupo in subgrupos:
        resultadoPorSubgrupo=resultadoAnalisis.loc[resultadoAnalisis['subgrupo'] == subgrupo]
        rentasAcumuladas=resultadoPorSubgrupo['rentaAcumulada']
        #Se toma la última fila del subgrupo para sacar la renta acumulada, y también el número de elementos
        ultimaRentaAcumulada=rentasAcumuladas.iloc[-1]
        numeroDiasAnalizados=len(rentasAcumuladas)
        resultadoPorSubgrupo.plot(kind='line', x='fecha', y='rentaAcumulada', ax=ax3, label=subgrupo+" --> "+'{:.1f}%'.format(ultimaRentaAcumulada)+', {:.0f}'.format(numeroDiasAnalizados)+' compras', marker="+")
    ax3.tick_params(axis='x', labelrotation=20)  # Rota las etiquetas del eje X
    formatter = mdates.DateFormatter("%Y-%m-%d")
    locator = mdates.DayLocator()
    plt.title('RENTA ACUM -espero ' + str(Y) + ' días SOLAPADOS-', fontsize=10)
    plt.xticks(rotation=90, ha='right')
    #plt.show()
    plt.savefig(dirAnalisis +"rentabilidadAcumulada-esperando " + str(Y) + " días.png")
    plt.close()

    # RENTABILIDAD MEDIA VS SP500
    ax4 = plt.gca()  # gca significa 'get current axis'
    for subgrupo in subgrupos:
        resultadoPorSubgrupo=resultadoAnalisis.loc[resultadoAnalisis['subgrupo'] == subgrupo]
        resultadoPorSubgrupo['rentaRelativaSP500']=resultadoPorSubgrupo['rentaRelativaSP500']/int(Y)
        media = np.mean(resultadoPorSubgrupo['rentaRelativaSP500'])
        numeroDiasAnalizados=len(resultadoPorSubgrupo['rentaRelativaSP500'])
        resultadoPorSubgrupo.plot(kind='line', x='fecha', y='rentaRelativaSP500', ax=ax4, label=subgrupo+" --> "+'{:.1f}%'.format(media)+', {:.0f}'.format(numeroDiasAnalizados)+' compras', marker="+")
    ax4.tick_params(axis='x', labelrotation=20)  # Rota las etiquetas del eje X
    formatter = mdates.DateFormatter("%Y-%m-%d")
    locator = mdates.DayLocator()
    plt.title('RENTA DIARIA vs SP500 por subgrupo y fecha', fontsize=10)
    plt.xticks(rotation=90, ha='right')
    #plt.show()
    plt.savefig(dirAnalisis +"rentabilidadvsSP500-esperando " + str(Y) + " días.png")
    plt.close()

    # RENTABILIDAD ACUMULADA
    ax5 = plt.gca()  # gca significa 'get current axis'
    for subgrupo in subgrupos:
        resultadoPorSubgrupo=resultadoAnalisis.loc[resultadoAnalisis['subgrupo'] == subgrupo]
        rentasAcumuladasvsSP500=resultadoPorSubgrupo['rentaAcumuladavsSP500']
        #Se toma la última fila del subgrupo para sacar la rentaAcumuladavsSP500, y también el número de elementos
        ultimaRentaAcumuladavsSP500=rentasAcumuladasvsSP500.iloc[-1]
        numeroDiasAnalizados=len(rentasAcumuladasvsSP500)
        resultadoPorSubgrupo.plot(kind='line', x='fecha', y='rentaAcumuladavsSP500', ax=ax5, label=subgrupo+" --> "+'{:.1f}%'.format(ultimaRentaAcumuladavsSP500)+', {:.0f}'.format(numeroDiasAnalizados)+' compras', marker="+")
    ax5.tick_params(axis='x', labelrotation=20)  # Rota las etiquetas del eje X
    formatter = mdates.DateFormatter("%Y-%m-%d")
    locator = mdates.DayLocator()
    plt.title('RENTA ACUM vs SP500 -espero ' + str(Y) + ' días SOLAPADOS-', fontsize=10)
    plt.xticks(rotation=90, ha='right')
    #plt.show()
    plt.savefig(dirAnalisis +"rentabilidadAcumuladavsSP500-esperando " + str(Y) + " días.png")
    plt.close()

    # RENTABILIDAD ACUMULADA CON PROB UNO MÍNIMA
    ax6 = plt.gca()  # gca significa 'get current axis'
    for subgrupo in subgrupos:
        resultadoPorSubgrupo=resultadoAnalisis.loc[resultadoAnalisis['subgrupo'] == subgrupo]
        rentasAcumuladasvsSP500ConProbUnoMinima=resultadoPorSubgrupo['rentaAcumuladavsSP500ConProbUnoMinima']
        #Se toma la última fila del subgrupo para sacar la rentasAcumuladasvsSP500ConProbUnoMinima, y también el número de elementos
        ultimaRentaAcumuladavsSP500ConProbUnoMinima=rentasAcumuladasvsSP500ConProbUnoMinima.iloc[-1]
        numeroDiasAnalizados=len(rentasAcumuladasvsSP500ConProbUnoMinima)
        resultadoPorSubgrupo.plot(kind='line', x='fecha', y='rentaAcumuladavsSP500ConProbUnoMinima', ax=ax6, label=subgrupo+" --> "+'{:.1f}%'.format(ultimaRentaAcumuladavsSP500ConProbUnoMinima)+', {:.0f}'.format(numeroDiasAnalizados)+' compras', marker="+")
    ax6.tick_params(axis='x', labelrotation=20)  # Rota las etiquetas del eje X
    formatter = mdates.DateFormatter("%Y-%m-%d")
    locator = mdates.DayLocator()
    plt.title('Prob MIN=' +'{:.2f}%'.format(probUnoMinima) +' -> RENTA ACUM vs SP500 -espero ' + str(Y) + ' días SOLAPADOS-', fontsize=10)
    plt.xticks(rotation=90, ha='right')
    #plt.show()
    plt.savefig(dirAnalisis +"rentaAcumuladavsSP500ConProbUnoMinima-esperando " + str(Y) + " días.png")
    plt.close()

#-----------------------------------------------------------
def analizar(datosGrandes, datosManejables, X, dfSP500):

    # Se obtiene una relación unívoca entre antigüedad y fecha. El valor 0 será el más reciente. Entonces, si quiero ir hacia el futuro, debo restar posiciones del índice
    # Se aplicará el mismo formato donde se necesite.
    fechasAcumuladas = 10000 * datosGrandes['anio'] + 100 * datosGrandes['mes'] + \
                                   datosGrandes['dia']
    fechasPorAntiguedad=fechasAcumuladas.unique()
    listaFechasPorAntiguedad=np.flip(np.sort(fechasPorAntiguedad)).tolist()

    # OBTENCIÓN DEL CLOSE PARA LAS FILAS PREDICHAS TRAS X DÍAS POSTERIORES
    datosMergeados = pd.merge(datosGrandes, datosManejables, how='right',
                              on=['empresa', 'anio', 'mes', 'dia', 'close', 'subgrupo'])
    #Se busca qué valor en fecha es la antigüedad desplazada
    datosMergeados['aniomesdia']= 10000 * datosMergeados['anio'] + 100 * datosMergeados['mes'] + \
      datosMergeados['dia']
    datosGrandes['aniomesdia'] = 10000 * datosGrandes['anio'] + 100 * datosGrandes['mes'] + \
                                 datosGrandes['dia']

    datosDesplazados=pd.DataFrame()
    for row_index, row in datosMergeados.iterrows():
        indice=listaFechasPorAntiguedad.index(row['aniomesdia'])
        # Desplazo el índice hacia el futuro. Es decir, resto
        indice-=int(X)
        #Sólo guardaré la fila si el índice es mayor o igual que 0
        if indice >= 0:
            fechaDesplazada=listaFechasPorAntiguedad[indice]
            fila=datosGrandes.loc[(datosGrandes['aniomesdia']==fechaDesplazada) & (datosGrandes['empresa']==row['empresa'])]
            # Se encuentra la misma empresa en varios subgrupos. Cogemos el primero, ya que buscamos el close, que es común
            # si no hay datos, saltamos
            if len(fila.index)>0:
                fila=fila.iloc[0]
                row['closeFuturo']=fila['close']
                datosDesplazados=datosDesplazados.append(row, ignore_index=True)


    datosFuturo = pd.merge(datosGrandes, datosDesplazados, how='right', on=['empresa', 'aniomesdia', 'subgrupo'])

    datosAAnalizar = datosFuturo.loc[datosFuturo['TARGET_x'].isin(
        ['1', '0'])]  # Son datos tan antiguos que sí tienen su resultado futuro (que es el REAL)


    # CÁLCULO DE RENDIMIENTO MEDIO POR FECHA Y SUBGRUPO
    # En fecha_x está el futuro. En fecha_y está el dato predicho. Se genera una columna nueva que obtiene el rendimiento real
    datosAAnalizar.loc[:, 'rendimiento'] = 100 * (datosAAnalizar['closeFuturo'] - datosAAnalizar['close_x']) / \
                                           datosAAnalizar['close_x']

    grupos = datosAAnalizar.groupby(['mes_y', 'dia_y', 'subgrupo'])

    resultadoAnalisis = pd.DataFrame(
        columns=["fecha", "anio", "mes", "dia", "subgrupo", "precisionMedia", "rentaMedia", "numElementos",
                 "rentaRelativaSP500"])

    # Se calculan probabilidades, y se loggean/grafican por antiguedad
    for group_name, df_group in grupos:
        # En cada antigüedad se reinician los contadores
        rentaMedia = 0
        rentasSubgrupo = []
        rentasSubgrupoConProbUnoMinima = []
        anio = 0
        mes = 0
        dia = 0
        numPrecisionAcertada = 0
        numPrecisionFallada = 0
        precisionMedia = -1

        for row_index, row in df_group.iterrows():
            rentasSubgrupo.append(row['rendimiento'])
            anio = row['anio_y']
            mes = row['mes_y']
            dia = row['dia_y']
            subgrupo = row['subgrupo']
            target = row['TARGET_y']
            if target == 1:
                numPrecisionAcertada += 1
            elif target == 0:
                numPrecisionFallada += 1

            # Selecciono sólo los que superen un umbral de probabilidad
            if (row['TARGET_PREDICHO_PROB'] > probUnoMinima):
                rentasSubgrupoConProbUnoMinima.append(row['rendimiento'])

        # Se calcula la precisión
        if (numPrecisionAcertada + numPrecisionFallada) > 0:
            precisionMedia = numPrecisionAcertada / (numPrecisionAcertada + numPrecisionFallada)

        # Se calcula la renta media
        rentaMedia = np.mean(rentasSubgrupo)

        # Se calcula la renta media con probab uno mínima
        rentaMediaConProbUnoMinima = np.mean(rentasSubgrupoConProbUnoMinima)

        # Se cuenta el número de elementos analizados por subgrupo
        numElementos = len(rentasSubgrupo)

        # Se guarda la fecha
        datetime_str = str(int(dia)) + '/' + str(int(mes)) + '/' + str(int(anio))
        fecha = datetime.strptime(datetime_str, '%d/%m/%Y')

        # Se calcula la renta media respecto al SP500
        filaSP500 = dfSP500.loc[dfSP500['fecha'] == fecha]
        rentaRelativaSP500 = float(rentaMedia - filaSP500['rentaSP500'])
        rentaRelativaSP500ConProbUnoMinima = float(rentaMediaConProbUnoMinima - filaSP500['rentaSP500'])

        # Se imprimen los resultados
        print('\n ANIO/MES/DIA: {:.0f}'.format(anio) + "/" + '{:.0f}'.format(mes) + "/" + '{:.0f}'.format(dia))
        print(' SUBGRUPO: {}'.format(subgrupo))
        print(' PRECISION MEDIA: {:.0f}'.format(numPrecisionAcertada) + '/' + '{:.0f}'.format(
            numPrecisionAcertada + numPrecisionFallada) + ' = {:.2f}'.format(precisionMedia))
        print(' NUMERO ELEMENTOS: {:.0f}'.format(numElementos))
        print(' RENTABILIDAD MEDIA: {:.2f}%'.format(rentaMedia))
        print(' RENTABILIDAD SP500: {:.2f}%'.format(float(filaSP500['rentaSP500'])))
        print(' RENTABILIDAD vs SP500: {:.2f}%'.format(rentaRelativaSP500))
        print(' RENTABILIDAD MEDIA CON PROB UNO MÍNIMA: {:.2f}%'.format(rentaMediaConProbUnoMinima))
        print(' RENTABILIDAD vs SP500 CON PROB UNO MÍNIMA: {:.2f}%'.format(rentaRelativaSP500ConProbUnoMinima))

        # Se guardan los resultados en un DataFrame
        nuevaFila = [{'fecha': fecha, 'anio': anio, 'mes': mes, 'dia': dia, 'subgrupo': subgrupo,
                      'precisionMedia': precisionMedia, 'rentaMedia': rentaMedia,
                      'numElementos': '{:.0f}'.format(numElementos), 'rentaRelativaSP500': rentaRelativaSP500,
                      'rentaMediaConProbUnoMinima': rentaMediaConProbUnoMinima,
                      'rentaRelativaSP500ConProbUnoMinima': rentaRelativaSP500ConProbUnoMinima}]
        resultadoAnalisis = resultadoAnalisis.append(nuevaFila, ignore_index=True, sort=False)

    # Separación por subgrupos
    resultadoAnalisis['aniomesdia'] = 10000 * resultadoAnalisis['anio'] + 100 * resultadoAnalisis['mes'] + \
                                      resultadoAnalisis['dia']
    subgrupos = resultadoAnalisis['subgrupo'].unique().tolist()

    # Se calcula la renta acumulada
    resultadoAnalisisAux = pd.DataFrame(
        columns=["fecha", "anio", "mes", "dia", "subgrupo", "precisionMedia", "rentaMedia", "numElementos",
                 "rentaRelativaSP500", "rentaAcumulada", "rentaAcumuladavsSP500", "rentaMediaConProbUnoMinima"])
    for subgrupo in subgrupos:
        resultadoPorSubgrupo = resultadoAnalisis.loc[resultadoAnalisis['subgrupo'] == subgrupo]
        resultadoPorSubgrupoAFecha = resultadoAnalisis.loc[resultadoAnalisis['subgrupo'] == subgrupo]['fecha']
        resultadoPorSubgrupoBFecha = resultadoAnalisis.loc[resultadoAnalisis['subgrupo'] == subgrupo]['fecha']
        for fechaA in resultadoPorSubgrupoAFecha:
            rentaAcumulada = 0
            rentaAcumuladaConProbUnoMinima = 0
            rentaAcumuladavsSP500 = 0
            rentaAcumuladavsSP500ConProbUnoMinima = 0
            resultadoACompleto = resultadoPorSubgrupo.loc[resultadoPorSubgrupo['fecha'] == fechaA]
            for fechaB in resultadoPorSubgrupoBFecha:
                resultadoBCompleto = resultadoPorSubgrupo.loc[resultadoPorSubgrupo['fecha'] == fechaB]
                if int(resultadoACompleto['aniomesdia']) >= int(resultadoBCompleto['aniomesdia']):
                    # Para los resultados anteriores o iguales en fecha, se calcula la rentaAcumulada y rentaAcumuladavsSP500
                    rentaAcumulada += float(resultadoBCompleto['rentaMedia'])
                    rentaAcumuladavsSP500 += float(resultadoBCompleto['rentaRelativaSP500'])
                    rentaAcumuladaConProbUnoMinima += float(resultadoBCompleto['rentaMediaConProbUnoMinima'])
                    rentaAcumuladavsSP500ConProbUnoMinima += float(
                        resultadoBCompleto['rentaRelativaSP500ConProbUnoMinima'])

            nuevaFilaAux = resultadoACompleto
            nuevaFilaAux.loc[:, 'rentaAcumulada'] = rentaAcumulada
            nuevaFilaAux.loc[:, 'rentaAcumuladavsSP500'] = rentaAcumuladavsSP500
            nuevaFilaAux.loc[:, 'rentaAcumuladaConProbUnoMinima'] = rentaAcumuladaConProbUnoMinima
            nuevaFilaAux.loc[:, 'rentaAcumuladavsSP500ConProbUnoMinima'] = rentaAcumuladavsSP500ConProbUnoMinima
            resultadoAnalisisAux = resultadoAnalisisAux.append(nuevaFilaAux, ignore_index=True, sort=False)
    resultadoAnalisis = resultadoAnalisisAux

    # Se escriben los resultados a un Excel
    pathCsvResultados = dirAnalisis + "RESULTADOS_ANALISIS" + str(X) + ".csv"
    print("Guardando: " + pathCsvResultados)
    resultadoAnalisis.to_csv(pathCsvResultados, index=False, sep='|')

    return resultadoAnalisis, subgrupos


#------------------------SP500-----------------------------

#Descarga del histórico del SP500
today = date.today()
fechaInicio="2019-01-01"
fechaFin=today.strftime("%Y-%m-%d")
url="https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&" \
    "graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1168&" \
    "nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=SP500&scale=left&cosd="+fechaInicio\
    +"&coed=fechaFin&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&" \
     "oet=99999&mma=0&fml=a&fq=Daily%2C%20Close&fam=avg&fgst=lin&fgsnd=2010-04-12&line_index=1&transformation=lin&" \
     "vintage_date="+fechaFin+"&revision_date=2020-04-10&nd=2010-04-12"
destino=dirAnalisis+"SP500.csv"
myfile = requests.get(url, allow_redirects=True)
open(destino, 'wb').write(myfile.content)

#Se guarda en dataframe
columnasSP500 = ['DATE', 'SP500']
datosSP500 = pd.read_csv(filepath_or_buffer=destino, sep=',')

#Sólo me quedo con las filas cuyo precio sea numérico
datosSP500 = datosSP500.loc[~(datosSP500['SP500'] == '.')]
# resetting index
datosSP500.reset_index(inplace = True)

#La fecha se convierte a dato de fecha
dfSP500=pd.DataFrame(columns=["fecha", "close", "rentaSP500"])
closeXDiasFuturos=0
tamaniodfSP500=len(datosSP500)
for index, fila in datosSP500.iterrows():
    fecha = datetime.strptime(fila['DATE'], '%Y-%m-%d')
    if index < (tamaniodfSP500 - int(X)):
        filaXDiasFuturos=datosSP500.iloc[index+int(X)]
        closeXDiasFuturos=filaXDiasFuturos ['SP500']
        rentaSP500 = 100*(float(closeXDiasFuturos)-float(fila['SP500']))/float(closeXDiasFuturos)
    else:
        rentaSP500 = 0

    nuevaFila = [{'fecha': fecha, 'close': float(fila['SP500']), 'rentaSP500':float(rentaSP500)}]
    dfSP500 = dfSP500.append(nuevaFila, ignore_index=True, sort=False)

# CLOSE SP500
ax4 = plt.gca()  # gca significa 'get current axis'
dfSP500.plot(kind='line', x='fecha', y='close', ax=ax4, label='SP500', marker="+")
ax4.tick_params(axis='x', labelrotation=20)  # Rota las etiquetas del eje X
formatter = mdates.DateFormatter("%Y-%m-%d")
locator = mdates.DayLocator()
plt.title('CLOSE SP500')
plt.xticks(rotation=90, ha='right')
plt.savefig(dirAnalisis+"SP500-close.png")
plt.close()

# RENTABILIDAD SP500 tras X días
ax4 = plt.gca()  # gca significa 'get current axis'
dfSP500.plot(kind='line', x='fecha', y='rentaSP500', ax=ax4, label='SP500', marker="+")
ax4.tick_params(axis='x', labelrotation=20)  # Rota las etiquetas del eje X
formatter = mdates.DateFormatter("%Y-%m-%d")
locator = mdates.DayLocator()
plt.title('RENTABILIDAD DIARIA SP500 en % - tras '+ X + ' días-')
plt.xticks(rotation=90, ha='right')
plt.savefig(dirAnalisis+"SP500-rentabilidad.png")
plt.close()

#---------------------------------RECOGIDA DE GRANDES_0 Y MANEJABLES-------------------------------

# Se listan los ficheros manejables y grandes (sólo los más recientes, con antigüedad 0 en el nombre)
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
    datosManejables=datosManejables.append(datosFicheroInteresantes, ignore_index=True)


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
        fechaTemporal=fechaFichero

for filename in ficherosGrandesCero:
    fechaFichero=int(filename[(filename.rfind('/')+1):(filename.rfind('/')+9)])
    #Se tomará sólo el fichero grande más reciente
    if fechaFichero==fechaTemporal:
        ficherosGrandesCeroAAnalizar.append(filename)

for filename in ficherosGrandesCeroAAnalizar:
    datosFichero = pd.read_csv(filepath_or_buffer=filename, sep='|')
    datosFicheroReducidos=datosFichero[columnasGrandes]
    datosFicheroInteresantes=pd.DataFrame(datosFicheroReducidos, columns=columnasGrandes)
    datosFicheroInteresantes['subgrupo'] = filename[(filename.rfind('SG') + 3):(filename.rfind('SG') + 5)].replace('_',
                                                                                                                   '')
    datosGrandes=datosGrandes.append(datosFicheroInteresantes)

#---------------------------------ANÁLISIS-------------------------------
resultadoAnalisis, subgrupos=analizar(datosGrandes, datosManejables, X, dfSP500)
resultadoAnalisisMenosDos, subgruposMenosDos=analizar(datosGrandes, datosManejables, int(X)-2, dfSP500)
resultadoAnalisisMasDos, subgruposMasDos=analizar(datosGrandes, datosManejables, int(X)+2, dfSP500)

#---------------------------------DIBUJOS-------------------------------
pintar(resultadoAnalisis, subgrupos, X)
pintar(resultadoAnalisisMenosDos, subgruposMenosDos, int(X)-2)
pintar(resultadoAnalisisMasDos, subgruposMasDos, int(X)+2)

print("\n--- InversionUtilsPosteriori: FIN ---")



