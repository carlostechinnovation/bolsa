import sys
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("\n********* Analisis de uso de las FEATURES por cada modelo ganador en todos los subgrupos ********* ")

print("PARAMETROS: ")
dir_subgrupos = sys.argv[1]
pathSalida = sys.argv[2]
print("dir_subgrupos: %s" % dir_subgrupos)
print("pathSalida: %s" % pathSalida)

acumuladoDF = pd.DataFrame(columns=['id_subgrupo', 'columnas_seleccionadas'])
contador=0
featuresTodas = list()

for directorio in os.listdir(dir_subgrupos):
    contador = contador + 1
    id_subgrupo = directorio.split("_")[1]

    #Extraer TODAS las features, mirando la cabecera del REDUCIDO
    pathFileReducido = dir_subgrupos + directorio + "/REDUCIDO.csv"
    f = open(pathFileReducido, "r")
    columnas = f.readline().split("|")
    columnasLimpio=[]
    for sub in columnas:
        columnasLimpio.append(sub.replace("\n", ""))

    featuresTodas.extend(filter(lambda x: x, columnasLimpio))  # añade los elementos, habiendo quitado los nulos y limpiado los saltos de carro
    print(featuresTodas)

    pathFicheroAbsoluto = dir_subgrupos + directorio + "/FEATURES_SELECCIONADAS.csv"
    if os.path.exists(pathFicheroAbsoluto):
        f = open(pathFicheroAbsoluto, "r")
        featuresModelo = f.read()
        print("id_subgrupo = " + id_subgrupo + " -> features= " + featuresModelo)
        subgrupoDF = pd.DataFrame(data=[[id_subgrupo, featuresModelo]], columns=['id_subgrupo', 'columnas_seleccionadas'])
        acumuladoDF = acumuladoDF.append(subgrupoDF)

#Limpiar nombres de columnas duplicados
featuresTodas = list(set(featuresTodas))

#### MATRIZ ####
columnas=featuresTodas
columnasConId = columnas.copy()
columnasConId.insert(0, 'id_subgrupo')
columnasConId.append('Numero de features usadas')
matrizDF = pd.DataFrame(columns=columnasConId)

for row in acumuladoDF.itertuples(index=False):
    idSubgrupo = row.id_subgrupo
    colSeleccionadas = row.columnas_seleccionadas.split("|")
    fila = []
    fila.append(int(idSubgrupo))  # ID de SUBGRUPO
    num_features = 0  # Numero de features usadas por este subgrupo
    for columna in columnas:
        if columna in colSeleccionadas:
            fila.append(1)
            num_features += 1
        else:
            fila.append(0)

    fila.insert(len(fila), num_features)
    filaDatos = pd.Series(fila, index=columnasConId)   # anhadir elemento primero (prepend)
    matrizDF = matrizDF.append(filaDatos, ignore_index=True)


# ESTILOS CSS
def pintarColores(val):
    if int(val) == 1:
        color = 'background-color: #64b41a99'
    elif 2 <= int(val) < 20:
        color = 'background-color: #f0f600b3'
    elif 20 <= int(val) < 50:
        color = 'background-color: #ea980ae6'
    elif int(val) >= 50:
        color = 'background-color: #18b054b3'
    else:
        color = ''

    return 'text-align: right; border: 1px solid black; border-collapse: collapse; %s' % color

#Sort por nombre de COLUMNAS Alfabeticamente

#Sort FILAS by id_subgrupo
matrizDF = matrizDF.sort_values(by=['id_subgrupo'])

# IMPORTANCIA DE LAS FEATURES: porcentaje de veces que cada feature es usada por los modelos
# Si queremos mejorar la métrica, debemos refinar esas features que las usan muchos modelos
matrizDF.loc['Porcentaje de uso'] = 100 * matrizDF.sum(axis=0)/matrizDF.count(axis=0)
matrizDF.at['Porcentaje de uso', 'id_subgrupo'] = 'Porcentaje de uso'  # Celda especial en la cabecera
matrizDF.at['Porcentaje de uso', 'Numero de features usadas'] = '0'  # TOTAL de totales: vacio

#TRANSPONEMOS LA MATRIZ para que ocupe menos visualmente
matrizDFtraspuesta = matrizDF.transpose()
# Ponemos la primera fila en la cabecera
new_header = matrizDFtraspuesta.iloc[0]
matrizDFtraspuesta = matrizDFtraspuesta[1:]
matrizDFtraspuesta.columns = new_header

print("matrizDFtraspuesta: " + str(matrizDFtraspuesta.shape[0]) + " x " + str(matrizDFtraspuesta.shape[1]))
# matrizDFtraspuesta.to_csv(pathSalida, index=False, sep='|')

print("Escribiendo en: " + pathSalida)
datosEnHtml = matrizDFtraspuesta.style.applymap(pintarColores).render()
text_file = open(pathSalida, "w")
text_file.write(datosEnHtml)
text_file.close()

