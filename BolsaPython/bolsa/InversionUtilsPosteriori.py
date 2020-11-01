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
from docutils.nodes import subscript

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
    if not resultadoAnalisis.empty:
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
            if numeroDiasAnalizados > 0:
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
            std = np.mean(resultadoPorSubgrupo['stdMedia'])
            numeroDiasAnalizados=len(resultadoPorSubgrupo['rentaMedia'])
            resultadoPorSubgrupo.plot(kind='line', x='fecha', y='rentaMedia', ax=ax2, label=subgrupo+" --> "+'{:.1f}%'.format(media)+', {:.0f}'.format(numeroDiasAnalizados)+' compras, '+'std: {:.1f}%'.format(std), marker="+")
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
            std = np.mean(resultadoPorSubgrupo['stdMedia'])
            rentasAcumuladas=resultadoPorSubgrupo['rentaAcumulada']
            # Se eliminan los nulos
            rentasAcumuladas = rentasAcumuladas.dropna()
            #Se toma la última fila del subgrupo para sacar la renta acumulada, y también el número de elementos
            numeroDiasAnalizados=len(rentasAcumuladas)
            if numeroDiasAnalizados>0:
                ultimaRentaAcumulada=rentasAcumuladas.iloc[numeroDiasAnalizados-1]
                ultimaRentaAcumuladaPorCompra=ultimaRentaAcumulada/numeroDiasAnalizados
                resultadoPorSubgrupo.plot(kind='line', x='fecha', y='rentaAcumulada', ax=ax3, label=subgrupo+" --> "+'{:.1f}%'.format(ultimaRentaAcumuladaPorCompra)+' por compra, '+'{:.0f}'.format(numeroDiasAnalizados)+' c, '+' std: '+'{:.1f}%'.format(std), marker="+")
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
            mediaPorCompra=media/numeroDiasAnalizados
            resultadoPorSubgrupo.plot(kind='line', x='fecha', y='rentaRelativaSP500', ax=ax4, label=subgrupo+" --> "+'{:.1f}%'.format(mediaPorCompra)+' por compra, '+'{:.0f}'.format(numeroDiasAnalizados)+" c", marker="+")
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
            # Se eliminan los nulos
            rentasAcumuladasvsSP500 = rentasAcumuladasvsSP500.dropna()
            #Se toma la última fila del subgrupo para sacar la rentaAcumuladavsSP500, y también el número de elementos
            numeroDiasAnalizados=len(rentasAcumuladasvsSP500)
            if numeroDiasAnalizados>0:
                ultimaRentaAcumuladavsSP500=rentasAcumuladasvsSP500.iloc[numeroDiasAnalizados-1]
                ultimaRentaAcumuladavsSP500PorCompra=ultimaRentaAcumuladavsSP500/numeroDiasAnalizados
                resultadoPorSubgrupo.plot(kind='line', x='fecha', y='rentaAcumuladavsSP500', ax=ax5, label=subgrupo+" --> "+'{:.1f}%'.format(ultimaRentaAcumuladavsSP500PorCompra)+', por compra, '+'{:.0f}'.format(numeroDiasAnalizados)+" c", marker="+")
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
            #Se eliminan los nulos
            rentasAcumuladasvsSP500ConProbUnoMinima=rentasAcumuladasvsSP500ConProbUnoMinima.dropna()
            #Se toma la última fila del subgrupo para sacar la rentasAcumuladasvsSP500ConProbUnoMinima, y también el número de elementos
            numeroDiasAnalizados=len(rentasAcumuladasvsSP500ConProbUnoMinima)
            if numeroDiasAnalizados>0:
                ultimaRentaAcumuladavsSP500ConProbUnoMinima=rentasAcumuladasvsSP500ConProbUnoMinima.iloc[numeroDiasAnalizados-1]
                ultimaRentaAcumuladavsSP500ConProbUnoMinimaPorCompra=ultimaRentaAcumuladavsSP500ConProbUnoMinima/numeroDiasAnalizados
                resultadoPorSubgrupo.plot(kind='line', x='fecha', y='rentaAcumuladavsSP500ConProbUnoMinima', ax=ax6, label=subgrupo+" --> "+'{:.1f}%'.format(ultimaRentaAcumuladavsSP500ConProbUnoMinimaPorCompra)+' por compra, '+'{:.0f}'.format(numeroDiasAnalizados)+" c", marker="+")
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
                              on=['empresa', 'anio', 'mes', 'dia', 'close'])
    #Se busca qué valor en fecha es la antigüedad desplazada
    datosMergeados['aniomesdia']= 10000 * datosMergeados['anio'] + 100 * datosMergeados['mes'] + \
      datosMergeados['dia']
    datosGrandes['aniomesdia'] = 10000 * datosGrandes['anio'] + 100 * datosGrandes['mes'] + \
                                 datosGrandes['dia']

    datosDesplazados=pd.DataFrame()
    for row_index, row in datosMergeados.iterrows():
        aniomesdia=row['aniomesdia']
        if aniomesdia in listaFechasPorAntiguedad:
            indice=listaFechasPorAntiguedad.index(aniomesdia)
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

    if 'empresa' in datosGrandes and 'empresa' in datosDesplazados:
        datosFuturo = pd.merge(datosGrandes, datosDesplazados, how='right', on=['empresa', 'aniomesdia'])

        datosAAnalizar = datosFuturo.loc[datosFuturo['TARGET_x'].isin(
            ['1', '0'])]  # Son datos tan antiguos que sí tienen su resultado futuro (que es el REAL)

        # CÁLCULO DE RENDIMIENTO MEDIO POR FECHA Y SUBGRUPO
        # En fecha_x está el futuro. En fecha_y está el dato predicho. Se genera una columna nueva que obtiene el rendimiento real
        datosAAnalizar.loc[:, 'rendimiento'] = 100 * (datosAAnalizar['closeFuturo'] - datosAAnalizar['close_x']) / \
                                               datosAAnalizar['close_x']

        grupos = datosAAnalizar.groupby(['mes_y', 'dia_y', 'subgrupo'])

        resultadoAnalisis = pd.DataFrame(
            columns=["fecha", "anio", "mes", "dia", "subgrupo", "precisionMedia", "rentaMedia", "stdMedia", "numElementos",
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

            #Varianza
            stdMedia=np.std(rentasSubgrupo)

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
            print(' VARIANZA: {:.2f}%'.format(stdMedia))
            print(' RENTABILIDAD SP500: {:.2f}%'.format(float(filaSP500['rentaSP500'])))
            print(' RENTABILIDAD vs SP500: {:.2f}%'.format(rentaRelativaSP500))
            print(' RENTABILIDAD MEDIA CON PROB UNO MÍNIMA: {:.2f}%'.format(rentaMediaConProbUnoMinima))
            print(' RENTABILIDAD vs SP500 CON PROB UNO MÍNIMA: {:.2f}%'.format(rentaRelativaSP500ConProbUnoMinima))

            # Se guardan los resultados en un DataFrame
            nuevaFila = [{'fecha': fecha, 'anio': anio, 'mes': mes, 'dia': dia, 'subgrupo': subgrupo,
                          'precisionMedia': precisionMedia, 'rentaMedia': rentaMedia, 'stdMedia': stdMedia,
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
            columns=["fecha", "anio", "mes", "dia", "subgrupo", "precisionMedia", "rentaMedia", "stdMedia", "numElementos",
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

    else:
        print("Los datos son INCOMPLETOS. Salimos...")
        #Se generan los siguientes datos sólo por completitud, pero son fakes
        resultadoAnalisis = pd.DataFrame(
        columns=["fecha", "anio", "mes", "dia", "subgrupo", "precisionMedia", "rentaMedia", "stdMedia", "numElementos",
                 "rentaRelativaSP500", "rentaAcumulada", "rentaAcumuladavsSP500", "rentaMediaConProbUnoMinima"])
        subgrupos=resultadoAnalisis

    return resultadoAnalisis, subgrupos

#----------------------------------------------------------
def calidadSubgrupos(resultadoAnalisis, subgrupos):
    subgruposPorCalidad = pd.DataFrame(
        columns=["subgrupo", "calidad"])
    if not resultadoAnalisis.empty:
        # Se combinará la media y la varianza de renta diaria, y el número de elementos para obtener un único valor con el que ordenar los
        # subgrupos. Se le dará un valor de CALIDAD a cada subgrupo
        for subgrupo in subgrupos:
            resultadoPorSubgrupo=resultadoAnalisis.loc[resultadoAnalisis['subgrupo'] == subgrupo]
            rentaRelativaSP500=resultadoPorSubgrupo['rentaRelativaSP500']
            mediaRenta=np.mean(rentaRelativaSP500)
            medianaRenta = np.median(rentaRelativaSP500)
            stdRenta=np.std(rentaRelativaSP500)
            numElementos=len(rentaRelativaSP500)

            # Fórmula:
            # CALIDAD = mediana renta diaria * log (1 + número de elementos)
            calidadMediana = medianaRenta * np.log10(1 + numElementos)
            calidadMediaStd =(mediaRenta-np.sqrt(stdRenta))*np.log10(1+numElementos)

            nuevoGrupo = [{'subgrupo': subgrupo, 'calidadMediana': calidadMediana, 'calidadMediaStd': calidadMediaStd}]
            subgruposPorCalidad = subgruposPorCalidad.append(nuevoGrupo, ignore_index=True, sort=False)
            subgruposPorCalidad=subgruposPorCalidad.sort_values("calidadMediana", ascending=False)

        # Se sacan por log
        for row_index, row in subgruposPorCalidad.iterrows():
            print("subgrupo: " + row['subgrupo'] + "  -> CALIDAD: " + str(row['calidad']))

        # Se escriben los resultados a un csv
        pathCsvResultados = dirAnalisis + "CALIDAD" + ".csv"
        print("Guardando: " + pathCsvResultados)
        subgruposPorCalidad.to_csv(pathCsvResultados, index=False, sep='|')

    return subgruposPorCalidad

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
    print("Fichero analizado: "+filename)
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
subgrupos=[]

# # Se leerán sólo los de la fecha más reciente de entre todos los ficheros grandes, para cada subgrupo
# # Cada día sólo salen unos subgrupos, no todos. Así que se buscará el fichero más reciente de cada subgrupo
# # Primero se saca la fecha más reciente de cada subgrupo
# for filename in ficherosGrandesCero:
#     subgrupo=filename[(filename.rfind('SG') + 3):(filename.rfind('SG') + 5)].replace('_',
#                                                                             '')
#     subgrupos.append(subgrupo)
# #Se hace "unique" de los subgrupos
# subgrupos=set(subgrupos)
#
# for subgrupo in subgrupos:
#     ficherosGrandesCeroPorSubgrupo=[k for k in ficherosGrandesCero if str("_0_SG_"+subgrupo+"_") in k]
#     #El nombre del fichero más reciente del subgrupo comienza por la fecha más grande. Es decir, debo ordenar los ficheros por nombre y coger el primero
#     ficherosGrandesCeroPorSubgrupo.sort(reverse=True)
#     ficherosGrandesCeroAAnalizar.append(ficherosGrandesCeroPorSubgrupo[0])
#
# for filename in ficherosGrandesCeroAAnalizar:
#     datosFichero = pd.read_csv(filepath_or_buffer=filename, sep='|')
#     datosFicheroReducidos=datosFichero[columnasGrandes]
#     datosFicheroInteresantes=pd.DataFrame(datosFicheroReducidos, columns=columnasGrandes)
#     datosFicheroInteresantes['subgrupo'] = filename[(filename.rfind('SG') + 3):(filename.rfind('SG') + 5)].replace('_',
#                                                                                                                    '')
#     datosGrandes=datosGrandes.append(datosFicheroInteresantes)

# SE COGE EL FICHERO GRANDE DE SG_0 CON LA FECHA MÁS RECIENTE. CONTENDRÁ LOS PRECIOS DE TODAS LAS EMPRESAS EN LA ÚLTIMA FECHA
ficherosGrandesCeroGrupo0 = glob.glob(dirDropbox+"*_GRANDE_0_SG_0_*")
# Se ordena por fecha, y se toma el más reciente
ficherosGrandesCeroGrupo0.sort(reverse=True)
ficherosGrandesCeroGrupo0MasReciente=ficherosGrandesCeroGrupo0[0]
datosFichero = pd.read_csv(filepath_or_buffer=ficherosGrandesCeroGrupo0MasReciente, sep='|')
datosFicheroReducidos=datosFichero[columnasGrandes]
datosFicheroInteresantes=pd.DataFrame(datosFicheroReducidos, columns=columnasGrandes)
datosGrandes=datosGrandes.append(datosFicheroInteresantes)


#------------------------------ANÁLISIS BÁSICO -----------------------------

resultadoAnalisis, subgrupos=analizar(datosGrandes, datosManejables, X, dfSP500)
pintar(resultadoAnalisis, subgrupos, X)

#------------------------------DÓNDE SE DEBE INVERTIR---------------------------
subgruposPorCalidad=calidadSubgrupos(resultadoAnalisis, subgrupos)

#------------------------------ANÁLISIS AVANZADO -----------------------------
resultadoAnalisisMenosDos, subgruposMenosDos=analizar(datosGrandes, datosManejables, int(X)-2, dfSP500)
pintar(resultadoAnalisisMenosDos, subgruposMenosDos, int(X)-2)

resultadoAnalisisMasDos, subgruposMasDos=analizar(datosGrandes, datosManejables, int(X)+2, dfSP500)
pintar(resultadoAnalisisMasDos, subgruposMasDos, int(X)+2)
resultadoAnalisisMasCuatro, subgruposMasCuatro=analizar(datosGrandes, datosManejables, int(X)+4, dfSP500)
pintar(resultadoAnalisisMasCuatro, subgruposMasCuatro, int(X)+4)
resultadoAnalisisMasSeis, subgruposMasSeis=analizar(datosGrandes, datosManejables, int(X)+6, dfSP500)
pintar(resultadoAnalisisMasSeis, subgruposMasSeis, int(X)+6)
resultadoAnalisisMasOcho, subgruposMasOcho=analizar(datosGrandes, datosManejables, int(X)+8, dfSP500)
pintar(resultadoAnalisisMasOcho, subgruposMasOcho, int(X)+8)
resultadoAnalisisMasDiez, subgruposMasDiez=analizar(datosGrandes, datosManejables, int(X)+10, dfSP500)
pintar(resultadoAnalisisMasDiez, subgruposMasDiez, int(X)+10)
resultadoAnalisisMasDoce, subgruposMasDoce=analizar(datosGrandes, datosManejables, int(X)+12, dfSP500)
pintar(resultadoAnalisisMasDoce, subgruposMasDoce, int(X)+12)


print("\n--- InversionUtilsPosteriori: FIN ---")



