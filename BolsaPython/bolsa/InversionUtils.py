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

###################################
print("Construimos PREFIJO con año-mes-dia del primer dato que vea...")
anio = int(filasFiltradas['anio'][0])
mes = int(filasFiltradas['mes'][0])
dia = int(filasFiltradas['dia'][0])
amd = 10000*anio + 100*mes + dia
prefijo=str(amd)
print(prefijo)

pathEntrada = dirSalida + prefijo + "_GRANDE_"+sufijoFicheroSalida
pathSalida = dirSalida + prefijo + "_MANEJABLE_"+sufijoFicheroSalida
print("pathEntrada = " + pathEntrada)
print("pathSalida = " + pathSalida)
datosEntrada.to_csv(pathEntrada, index=False, sep='|')
filasFiltradas.to_csv(pathSalida, index=False, sep='|')

print("--- InversionUtils: FIN ---")
