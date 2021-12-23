import sys

import pandas as pd
from tabulate import tabulate

print("Pintar metricas y rentabilidades - INICIO")
##################################################################################################
print("PARAMETROS: ")
pathMetricasPasadoEntrada = sys.argv[1]
pathHtmlSalida = sys.argv[2]

print("pathMetricasPasadoEntrada = %s" % pathMetricasPasadoEntrada)
print("pathHtmlSalida = %s" % pathHtmlSalida)


################################# FUNCIONES #######################################################
def color_negative_red(val):
    float_value = None
    try:
        float_value = float(val)
    except ValueError:
        return None

    condicion = (float(float_value) >= 0.20 and float(float_value) > 0 and float(float_value)<=1)
    color = 'green' if condicion else 'black'
    bgcolor = 'lightgreen' if condicion else 'white'
    return 'color: %s; background-color: %s' % (color, bgcolor)


##################################################################################################

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
metricasEntrada = metricasEntrada.applymap(lambda a: a.replace('id_subgrupo:SG_', '').replace('precisionpasadotrain:', '').replace('precisionpasadotest:', '').replace('precisionpasadovalidacion:', ''). replace('precisionsistemarandom:', ''). replace('mejoraRespectoSistemaRandom:', ''))

######################################################

metricasEntradaHtml = metricasEntrada.style.applymap(color_negative_red).render(index=False)
# metricasEntradaHtml = metricasEntrada.to_html(header=True, index=False)
metricasEntradaHtml = metricasEntradaHtml.replace("<table", "<table style=\"border: 1px solid black;\"")
metricasEntradaHtml = metricasEntradaHtml.replace("<td", "<td style=\"border: 1px solid black;\"")
metricasEntradaHtml = metricasEntradaHtml.replace("class=\"row_heading", "style=\"color:white\" class=\"row_heading")  # ocultar indices
style="color:white"

# print(metricasEntradaHtml)

print("Limpio:")
print(tabulate(metricasEntrada.head(), headers='keys', tablefmt='psql'))
f.write(metricasEntradaHtml)


f.write("<h2>PASADO - RENTABILIDADES (usando graficas y CALIDAD.csv):</h2>")
f.write("<p>Pendiente...</p>")

f.write("</body></html>")  # HTML cierre
f.close()  # Se cierra fichero
print("Pintar metricas y rentabilidades - FIN")
