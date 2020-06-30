import sys
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("\n********* Analisis de uso de las FEATURES por cada modelo ganador en todos los subgrupos ********* ")

print("PARAMETROS: ")
dir_subgrupos = sys.argv[1]
pathSalida = sys.argv[2]
print("dir_subgrupos: %s" % dir_subgrupos)
print("pathSalida: %s" % pathSalida)

acumuladoDF = pd.DataFrame(columns=['id_subgrupo', 'columnas_seleccionadas'])
contador=0
featuresTodas=""

for directorio in os.listdir(dir_subgrupos):
    contador = contador + 1
    id_subgrupo = directorio.split("_")[1]

    #Extraer TODAS las features, mirando la cabecera del COMPLETO
    pathFileCompleto = dir_subgrupos + directorio + "/COMPLETO.csv"
    if contador == 1:
        f = open(pathFileCompleto, "r")
        featuresTodas = f.readline()
        print(featuresTodas)


    pathFicheroAbsoluto = dir_subgrupos + directorio + "/FEATURES_SELECCIONADAS.csv"
    if os.path.exists(pathFicheroAbsoluto):
        f = open(pathFicheroAbsoluto, "r")
        featuresModelo = f.read()
        print("id_subgrupo = " + id_subgrupo + " -> features= " + featuresModelo)
        subgrupoDF = pd.DataFrame(data=[[id_subgrupo, featuresModelo]], columns=['id_subgrupo', 'columnas_seleccionadas'])
        acumuladoDF = acumuladoDF.append(subgrupoDF)


#### MATRIZ ####
columnas=featuresTodas.split("|")
columnasConId=columnas.copy()
columnasConId.insert(0, 'ID_SUBGRUPO')
matrizDF = pd.DataFrame(columns=columnasConId)

for row in acumuladoDF.itertuples(index=False):
    idSubgrupo=row.id_subgrupo
    colSeleccionadas=row.columnas_seleccionadas.split("|")
    fila = []
    fila.append(idSubgrupo)
    for columna in columnas:

        if columna in colSeleccionadas:
            fila.append(1)
        else:
            fila.append(0)

      # anhadir elemento primero (prepend)
    filaDatos = pd.Series(fila, index=columnasConId)
    matrizDF = matrizDF.append(filaDatos, ignore_index=True)



# ESTILOS CSS
def pintarColores(val):
    color = 'red' if val==1 else 'black'
    return 'background-color: %s' % color

matrizDF.style.applymap(pintarColores)

matrizDF.style.background_gradient(cmap='Blues').hide_index()

#TRANSPONEMOS LA MATRIZ para que ocupe menos visualmente
matrizDF = matrizDF.transpose()

print("matrizDF: " + str(matrizDF.shape[0]) + " x " + str(matrizDF.shape[1]))
# matrizDF.to_csv(pathSalida, index=False, sep='|')

datosEnHtml = matrizDF.to_html()
text_file = open(pathSalida, "w")
text_file.write(datosEnHtml)
text_file.close()







