import sys

import pandas as pd
from tabulate import tabulate

print("Pintar metricas y rentabilidades - INICIO")
##################################################################################################
print("PARAMETROS: ")
pathMetricasPasadoEntrada = sys.argv[1]
pathAciertosPasadoEntrada = sys.argv[2]
pathHtmlSalida = sys.argv[3]

print("pathMetricasPasadoEntrada = %s" % pathMetricasPasadoEntrada)
print("pathAciertosPasadoEntrada = %s" % pathAciertosPasadoEntrada)
print("pathHtmlSalida = %s" % pathHtmlSalida)


################################# FUNCIONES #######################################################
def color_negative_red(val):
    float_value = None
    UMBRAL1=0.25
    UMBRAL2=0.40
    try:
        float_value = float(val)
    except ValueError:
        return None

    condicion1 = (float(float_value) >= UMBRAL1 and float(float_value) < UMBRAL2 and float(float_value) > 0 and float(float_value)<=1)
    condicion2 = (float(float_value) >= UMBRAL2 and float(float_value) > 0 and float(float_value)<=1)

    color = 'black'
    bgcolor = 'white'
    if condicion1:
        color = 'green'
        bgcolor = 'lightgreen'
    elif condicion2:
        color = 'darkgreen'
        bgcolor = 'greenyellow'

    return 'color: %s; background-color: %s' % (color, bgcolor)


def color_aciertos(val):
    float_value = None
    UMBRAL1=30
    UMBRAL2=100
    try:
        float_value = float(val)
    except ValueError:
        return None

    condicion1 = (float(float_value) >= UMBRAL1 and float(float_value) < UMBRAL2 )
    condicion2 = (float(float_value) >= UMBRAL2)

    color = 'black'
    bgcolor = 'white'
    if condicion1:
        color = 'green'
        bgcolor = 'lightgreen'
    elif condicion2:
        color = 'darkgreen'
        bgcolor = 'greenyellow'

    return 'color: %s; background-color: %s' % (color, bgcolor)


################# METRICAS ###########################################################################
f = open(pathHtmlSalida, mode="w")  # Se abre fichero en modo overwrite

f.write("<!DOCTYPE html><head><style></style></head><html><body>")  # HTML apertura
f.write("<h2>PASADO - METRICA (precisión):</h2>")
print("Cargar datos (CSV)...")
metricasEntrada = pd.read_csv(filepath_or_buffer=pathMetricasPasadoEntrada, sep='|', header=None)
print("metricasEntrada (LEIDO): " + str(metricasEntrada.shape[0]) + " x " + str(metricasEntrada.shape[1]))
print(tabulate(metricasEntrada.head(), headers='keys', tablefmt='psql'))

# limpieza
metricasEntrada.columns = ['tipo', 'id_subgrupo', 'precision pasado train', 'precision pasado test',
                           'precision pasado validacion', 'precision sistema random', 'mejora respecto sistema random']
metricasEntrada.drop("tipo", axis=1, inplace=True)
metricasEntrada = metricasEntrada.applymap(lambda a: a.replace('id_subgrupo:SG_', '') \
                  .replace('precisionpasadotrain:', '').replace('precisionpasadotest:', '') \
                  .replace('precisionpasadovalidacion:', ''). replace('precisionsistemarandom:', '') \
                  .replace('mejoraRespectoSistemaRandom:', ''))

metricasEntradaHtml = metricasEntrada.style.applymap(color_negative_red).render(index=False)
metricasEntradaHtml = metricasEntradaHtml.replace("<table", "<table style=\"border: 1px solid black;\"")
metricasEntradaHtml = metricasEntradaHtml.replace("<td", "<td style=\"border: 1px solid black;\"")
metricasEntradaHtml = metricasEntradaHtml.replace("class=\"row_heading", "style=\"color:white\" class=\"row_heading")  # ocultar indices
print("Limpio:")
print(tabulate(metricasEntrada.head(), headers='keys', tablefmt='psql'))
f.write(metricasEntradaHtml)

################# ACIERTOS ###########################################################################
f.write("<h2>PASADO - ACIERTOS:</h2>")
print("Cargar aciertos (CSV)...")
aciertosEntrada = pd.read_csv(filepath_or_buffer=pathAciertosPasadoEntrada, sep='|', header=None)
print("aciertosEntrada (LEIDO): " + str(aciertosEntrada.shape[0]) + " x " + str(aciertosEntrada.shape[1]))
print(tabulate(aciertosEntrada.head(), headers='keys', tablefmt='psql'))

# limpieza
aciertosEntrada.columns = ['tipo', 'id_subgrupo', 'escenario', 'positivos reales',
                           'positivos predichos', 'aciertos']
aciertosEntrada.drop("tipo", axis=1, inplace=True)
aciertosEntrada = aciertosEntrada.applymap(lambda a: a.replace('id_subgrupo:SG_', '') \
                  .replace('escenario:', '').replace('positivosreales:', '') \
                  .replace('positivospredichos:', ''). replace('aciertos:', ''))

aciertosEntradaHtml = aciertosEntrada.style.applymap(color_aciertos).render(index=False)
aciertosEntradaHtml = aciertosEntradaHtml.replace("<table", "<table style=\"border: 1px solid black;\"")
aciertosEntradaHtml = aciertosEntradaHtml.replace("<td", "<td style=\"border: 1px solid black;\"")
aciertosEntradaHtml = aciertosEntradaHtml.replace("class=\"row_heading", "style=\"color:white\" class=\"row_heading")  # ocultar indices
print("Limpio:")
print(tabulate(aciertosEntrada.head(), headers='keys', tablefmt='psql'))
f.write(aciertosEntradaHtml)
######################################################

f.write("<h2>PASADO - RENTABILIDADES (usando graficas y CALIDAD.csv):</h2>")
f.write("<p>Pendiente...</p>")

f.write("</body></html>")  # HTML cierre
f.close()  # Se cierra fichero
print("Pintar metricas y rentabilidades - FIN")
