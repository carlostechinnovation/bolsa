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

#Cogemos solo las filas con la antiguedad indicada
antiguedad_int = int(antiguedad)
filasFiltradas = datosEntrada[datosEntrada['antiguedad'] == antiguedad_int]


pathSalida = dirSalida+"PENDIENTE"+sufijoFicheroSalida
print("pathSalida = " + pathSalida)
filasFiltradas.to_csv(pathSalida, index=False, sep='|')

print("--- InversionUtils: FIN ---")
