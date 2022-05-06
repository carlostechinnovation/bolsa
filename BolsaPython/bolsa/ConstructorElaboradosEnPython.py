import glob
import os
import shutil
import sys
import pandas as pd
from tabulate import tabulate
import ElaboradosUtils

##################### EXPLICACION #####################
# A partir de un dataFrame de features (algunas de ellas son identificadores), genera las columnas elaboradas. Una de ellas es el TARGET.
# Entrada: dataframe con columnas identificadoras y otras dinámicas.
# Salida: tiene las mismas columnas que las de entrada y además, las elaboradas. Si es del futuro, no tiene columna TARGET.
###############################################################

# Ejemplo:
# /bolsa/pasado/limpios/ /bolsa/pasado/elaborados/ 10 7 15 2 4 15 5.0 0.0 0 0


print("=========== CONSTRUCTOR ELABORADOS EN PYTHON: inicio ==============")

############################## PARAMETROS ####################################################################
print("PARAMETROS: directorioIn directorioOut S X R M F B umbralSubidaPorVela umbralMinimoGranVela filtroDinamico1 filtroDinamico2")
directorioIn = sys.argv[1]          # String
directorioOut = sys.argv[2]         # String
S = int(sys.argv[3])    # Integer
X = int(sys.argv[4])    # Integer
R = int(sys.argv[5])    # Integer
M = int(sys.argv[6])    # Integer
F = int(sys.argv[7])    # Integer
B = int(sys.argv[8])    # Integer
umbralSubidaPorVela = float(sys.argv[9])    # Float
umbralMinimoGranVela = float(sys.argv[10])  # Float
filtroDinamico1 = int(sys.argv[11])     # Integer
filtroDinamico2 = int(sys.argv[12])     # Integer

NUMERO_EMPRESAS_ANALIZAR_PROFILING = 0

########## MAIN ########################################################################################
if __name__ == '__main__':
    print("Numero de parametros de entrada: " + str(len(sys.argv)))
    for idx, arg in enumerate(sys.argv):
       print("El parametro #{} es {}".format(idx, arg))


########################################
print("Lista de ficheros de ENTRADA...")
obj = os.scandir()
entradasCsv = []

# dirs=directories
for (root, dirs, file) in os.walk(directorioIn):
    for f in file:
        if '.csv' in f:
            entradasCsv.append(f)

print("Hay " + str(len(entradasCsv)) + " ficheros en el directorio de entrada " + directorioIn)

########################################
modoTiempo = "futuro"  #default
if "pasado" in directorioIn:
    modoTiempo = "pasado"

#Limpieza
import glob

ficherosPrevios = glob.glob("/bolsa/"+modoTiempo+"/elaborados/*")
for f in ficherosPrevios:
    os.remove(f)

contador = 0
for f in entradasCsv:
    contador += 1
    analizarEntrada = (contador <= NUMERO_EMPRESAS_ANALIZAR_PROFILING)  #Permite analizar detalladamente la entrada si se desea
    ElaboradosUtils.procesarCSV(directorioIn + f, directorioOut + f, modoTiempo, analizarEntrada, S, X, R, M, F, B, umbralSubidaPorVela, umbralMinimoGranVela)
    #break ###TEMPORAL


print("=========== CONSTRUCTOR ELABORADOS EN PYTHON: fin ==============")

