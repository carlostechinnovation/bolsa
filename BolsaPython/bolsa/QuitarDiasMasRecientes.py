import sys

import pandas as pd

# EXPLICACION
# Se toma el fichero de entrada (podría ser el COMPLETO.csv de un subgrupo,
# y se eliminan todas las filas de x fechas más recientes -parametrizable-).
# Para ello, se desplazan todas las antigüedades esa cantidad (se resta la antigüedad actual menos el
# desplazamiento), y se eliminan todas las filas de antigüedad negativa.
# El resultado se guarda con el mismo nombre que el fichero leído (es decir, se sobreescribe).

print("--- QuitarDiasMasRecientes: INICIO ---")
pathFicheroEntrada = sys.argv[1]
diasRecientesAEliminar = sys.argv[2]
dirSalida = sys.argv[3]
nombreFicheroSalida = sys.argv[4]

print("pathFicheroEntrada = " + pathFicheroEntrada)
print("diasRecientesAEliminar = " + diasRecientesAEliminar)
print("dirSalida = " + dirSalida)
print("nombreFicheroSalida = " + nombreFicheroSalida)

datosEntrada = pd.read_csv(filepath_or_buffer=pathFicheroEntrada, sep='|')

print("Se desplazan todas las antigüedades del fichero...")
diasRecientesAEliminar_int = int(diasRecientesAEliminar)
datosEntrada['antiguedad'] = datosEntrada['antiguedad'] - diasRecientesAEliminar_int

print("Se eliminan todas las filas con antigüedad negativa...")

filasFiltradas = datosEntrada[datosEntrada['antiguedad'] >= 0]

print("El resultado de quitar días con antigüedades negativas se almacena en: " + dirSalida + nombreFicheroSalida)
filasFiltradas.to_csv(dirSalida + nombreFicheroSalida, index=False, sep='|')

print("--- QuitarDiasMasRecientes: FIN ---")
