import sys
import os
import pandas as pd

print("--- InversionUtils: INICIO ---")
pathFicheroEntrada = sys.argv[1]
antiguedad = sys.argv[2]
dirSalida = sys.argv[3]
sufijoFicheroSalida = sys.argv[4]

print("pathFicheroEntrada = " + pathFicheroEntrada)
print("antiguedad = " + antiguedad)
print("dirSalida = " + dirSalida)
print("sufijoFicheroSalida = " + sufijoFicheroSalida)

datosEntrada = pd.read_csv(filepath_or_buffer=pathFicheroEntrada, sep='|')

print("Quitamos COLUMNAS que ya nos estorban...")
datosEntrada = datosEntrada.drop(datosEntrada.filter(regex='PENDIENTE').columns, axis=1)
datosEntrada = datosEntrada.drop(datosEntrada.filter(regex='RATIO').columns, axis=1)
datosEntrada = datosEntrada.drop(datosEntrada.filter(regex='CURTOSIS').columns, axis=1)
datosEntrada = datosEntrada.drop(datosEntrada.filter(regex='SKEWNESS').columns, axis=1)

print("Cogemos solo las FILAS con la antiguedad indicada y con TARGET PREDICHO=1...")
antiguedad_int = int(antiguedad)
filasFiltradas = datosEntrada[datosEntrada['antiguedad'] == antiguedad_int]
filasFiltradas = filasFiltradas[filasFiltradas['TARGET_PREDICHO'] == 1]

num_filas = filasFiltradas.shape[0]
if(num_filas == 0):
    print("Hay 0 filas interesantes para invertir. Salimos...")

else:
    print("Hay " + str(num_filas) + " filas interesantes para invertir. Seguimos...")
    print("Construimos PREFIJO con año-mes-dia del primer dato que vea...")
    anio = filasFiltradas['anio'].head(1).values[0]
    mes = filasFiltradas['mes'].head(1).values[0]
    dia = filasFiltradas['dia'].head(1).values[0]
    amd = 10000*anio + 100*mes + dia
    prefijo = str(amd)
    print(prefijo)

    pathEntrada = dirSalida + prefijo + "_GRANDE_"+sufijoFicheroSalida
    pathSalida = dirSalida + prefijo + "_MANEJABLE_"+sufijoFicheroSalida
    print("pathEntrada = " + pathEntrada)
    print("pathSalida = " + pathSalida)
    datosEntrada.to_csv(pathEntrada, index=False, sep='|')
    filasFiltradas.to_csv(pathSalida, index=False, sep='|')

print("--- InversionUtils: FIN ---")
