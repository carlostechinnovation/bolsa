import sys
import os
import pandas as pd
from pathlib import Path
from pandas import DataFrame

print("TestIntegracionUtilidad1 - INICIO")
print("PARAMETROS: ")
pathFicheroIzdo = sys.argv[1]
pathFicheroDrcho = sys.argv[2]
empresa = sys.argv[3]
pathSalida = sys.argv[4]

print("pathFicheroIzdo = %s" % pathFicheroIzdo)
print("pathFicheroDrcho = %s" % pathFicheroDrcho)
print("empresa = %s" % empresa)
print("pathSalida = %s" % pathSalida)

if os.path.exists(pathFicheroIzdo) and os.path.exists(pathFicheroDrcho):
    fileIzdo = pd.read_csv(filepath_or_buffer=pathFicheroIzdo, sep='|', index_col=0)
    fileDcho = pd.read_csv(filepath_or_buffer=pathFicheroDrcho, sep='|', index_col=0)

    fileDcho = fileDcho[fileDcho['empresa'] == empresa]

    # CRUCE: fileIzdo  INNER JOIN (fileDcho WHERE empresa analizada) ON (indices)
    # En el lado derecho quitamos todas las columnas que no haya en el lazo izquierdo, para evitar que se creen columnas con sufijos "_x" y "_y"
    cruce=pd.merge(fileIzdo, fileDcho[fileDcho.columns.difference(fileIzdo.columns)], left_index=True, right_index=True)

    # Guardar
    cruce.to_csv(pathSalida, index=True, sep='|')


print("TestIntegracionUtilidad1 - FIN")