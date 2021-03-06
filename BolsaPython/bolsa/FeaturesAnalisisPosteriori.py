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
pathSalidaAntesDePca = sys.argv[2]
pathSalida = sys.argv[3]
print("dir_subgrupos: %s" % dir_subgrupos)
print("pathSalidaAntesDePca: %s" % pathSalidaAntesDePca)
print("pathSalida: %s" % pathSalida)

##################### FUNCION - ESTILOS CSS ####################################
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

################################################################################################
acumuladoElegidasDF = pd.DataFrame(columns=['id_subgrupo', 'columnas_elegidas'])
acumuladoDF = pd.DataFrame(columns=['id_subgrupo', 'columnas_seleccionadas'])
contador = 0
featuresTodasCompleto = list()
featuresElegidasRfecv = list()
featuresTodasReducido = list()

for directorio in os.listdir(dir_subgrupos):
    contador = contador + 1
    id_subgrupo = directorio.split("_")[1]

    # COMPLETO
    pathFileCompleto = dir_subgrupos + directorio + "/COMPLETO.csv"
    print("Leyendo las features del fichero COMPLETO (antes de PCA): " + pathFileCompleto)
    if os.path.exists(pathFileCompleto):
        f = open(pathFileCompleto, "r")
        columnas = f.readline().split("|")
        columnasCompleto = []
        for sub in columnas:
            columnasCompleto.append(sub.replace("\n", ""))

        featuresTodasCompleto.extend(filter(lambda x: x, columnasCompleto))

    # ELEGIDAS (con RFE-CV)
    pathFileElegidasRfecv = dir_subgrupos + directorio + "/FEATURES_ELEGIDAS_RFECV.csv"
    print("Leyendo las features del fichero ELEGIDAS (antes de PCA): " + pathFileCompleto)
    if os.path.exists(pathFileElegidasRfecv):
        f = open(pathFileElegidasRfecv, "r")
        columnas = f.readline().split("|")  # solo la primera linea
        columnasElegidas = []
        for sub in columnas:
            columnasElegidas.append(sub.replace("\n", ""))

        featuresElegidasRfecv=[]
        featuresElegidasRfecv.extend(filter(lambda x: x, columnasElegidas))
        subgrupoElegidasDF = pd.DataFrame(data=[[id_subgrupo, featuresElegidasRfecv]], columns=['id_subgrupo', 'columnas_elegidas'])
        acumuladoElegidasDF = acumuladoElegidasDF.append(subgrupoElegidasDF)

    #Extraer TODAS las features, mirando la cabecera del REDUCIDO
    pathFileReducido = dir_subgrupos + directorio + "/REDUCIDO.csv"
    print("Leyendo las features del fichero REDUCIDO (tras PCA): " + pathFileReducido)
    if os.path.exists(pathFileReducido):
        f = open(pathFileReducido, "r")
        columnas = f.readline().split("|")
        columnasLimpio = []
        for sub in columnas:
            columnasLimpio.append(sub.replace("\n", ""))

        featuresTodasReducido.extend(filter(lambda x: x, columnasLimpio))

        pathFicheroAbsoluto = dir_subgrupos + directorio + "/FEATURES_SELECCIONADAS.csv"
        print("Leyendo las features del fichero final (SELECCIONADAS tras PCA): " + pathFicheroAbsoluto)
        if os.path.exists(pathFicheroAbsoluto):
            f = open(pathFicheroAbsoluto, "r")
            featuresModelo = f.read()
            print("id_subgrupo = " + id_subgrupo + " -> features= " + featuresModelo)
            subgrupoDF = pd.DataFrame(data=[[id_subgrupo, featuresModelo]], columns=['id_subgrupo', 'columnas_seleccionadas'])
            acumuladoDF = acumuladoDF.append(subgrupoDF)


#Limpiar nombres de columnas duplicados
featuresTodasCompleto = list(set(featuresTodasCompleto))
featuresTodasReducido = list(set(featuresTodasReducido))

##################################### DESPUES DE PCA ####################################################################
#### MATRIZ ####
columnas = featuresTodasReducido
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

print("Ordenando por indice...")
matrizDFtraspuesta.sort_index(inplace=True)

print("matrizDFtraspuesta: " + str(matrizDFtraspuesta.shape[0]) + " x " + str(matrizDFtraspuesta.shape[1]))
# matrizDFtraspuesta.to_csv(pathSalida, index=False, sep='|')

print("Escribiendo en: " + pathSalida)
datosEnHtml = matrizDFtraspuesta.style.applymap(pintarColores).render()
text_file = open(pathSalida, "w")
text_file.write(datosEnHtml)
text_file.close()

##################################### ANTES DE PCA ####################################################################
columnas = featuresTodasCompleto
columnasConId = columnas.copy()
columnasConId.insert(0, 'id_subgrupo')
columnasConId.append('Numero de features usadas')
matrizDF = pd.DataFrame(columns=columnasConId)

for row in acumuladoElegidasDF.itertuples(index=False):
    idSubgrupo = row.id_subgrupo
    colElegidas = row.columnas_elegidas #.split("|")
    fila = []
    fila.append(int(idSubgrupo))  # ID de SUBGRUPO
    num_features = 0  # Numero de features usadas por este subgrupo
    for columna in columnas:
        if columna in colElegidas:
            fila.append(1)
            num_features += 1
        else:
            fila.append(0)

    fila.insert(len(fila), num_features)
    filaDatos = pd.Series(fila, index=columnasConId)   # anhadir elemento primero (prepend)
    matrizDF = matrizDF.append(filaDatos, ignore_index=True)

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

print("Ordenando por el nombre de campo...")
matrizDFtraspuesta.sort_index(inplace=True)  # NOMBRE DEL INDICE: nombre de campo
print("Ordenando por el porcentaje de uso...")
matrizDFtraspuesta=matrizDFtraspuesta.fillna("")
matrizDFtraspuesta = matrizDFtraspuesta.astype('float').sort_values(by=['Porcentaje de uso'], ascending=False)

print("Ordenando fila especial...")
indice = matrizDFtraspuesta.index.tolist()
filaEspecial=matrizDFtraspuesta.filter(like="Numero de features usadas", axis=0)
matrizDFtraspuesta=matrizDFtraspuesta.drop("Numero de features usadas")
matrizDFtraspuesta=matrizDFtraspuesta.append(filaEspecial)  # la ponemos al final

print("matrizDFtraspuesta: " + str(matrizDFtraspuesta.shape[0]) + " x " + str(matrizDFtraspuesta.shape[1]))
# matrizDFtraspuesta.to_csv(pathSalida, index=False, sep='|')

print("Escribiendo en: " + pathSalidaAntesDePca)
datosEnHtml = matrizDFtraspuesta.style.applymap(pintarColores).render()
text_file = open(pathSalidaAntesDePca, "w")
text_file.write(datosEnHtml)
text_file.close()

