import sys
import os
import pandas as pd

print("--- Validador en Python: INICIO ---")
dir_validacion = sys.argv[1]
print("Directorio que contiene csv_fut1 y csv_fut2 es: " + dir_validacion)
ficheros = sorted(os.listdir(dir_validacion), key=len)
ficheros_csv = []
for fichero in ficheros:
    if fichero.endswith("csv"):
        ficheros_csv.append(fichero)

print("Ordenar para saber cuál es fut1 y cuál es fut2.")
subgrupo="ERROR"
if len(ficheros_csv) == 2:
    fichero_csv_a = ficheros_csv[0]
    fichero_csv_b = ficheros_csv[1]
    a = int(fichero_csv_a.split("_")[0])
    b = int(fichero_csv_b.split("_")[0])

    subgrupo = fichero_csv_a.split("_")[1] + "_" + fichero_csv_a.split("_")[2]

    if a < b:
        csv_fut1 = dir_validacion + fichero_csv_b
        csv_fut2 = dir_validacion + fichero_csv_a
    else:
        csv_fut1 = dir_validacion + fichero_csv_a
        csv_fut2 = dir_validacion + fichero_csv_b
else:
    raise NameError("La carpeta de validacion deberia tener 2 ficheros CSV (fut1 y fut2) para poder compararlos: " + dir_validacion)

print("csv_fut1 = " + csv_fut1)
print("csv_fut2 = " + csv_fut2)
datos_fut1 = pd.read_csv(filepath_or_buffer=csv_fut1, sep='|')
datos_fut2 = pd.read_csv(filepath_or_buffer=csv_fut2, sep='|')

# FUTURO_1: cogemos las predicciones target=1
filas_con_futuro_predicho = datos_fut1['TARGET_PREDICHO'] == 1
datos_fut1 = datos_fut1[filas_con_futuro_predicho]
fut1_fila1 = datos_fut1.head(1)

fut1_anio = int(fut1_fila1['anio'])
fut1_mes = int(fut1_fila1['mes'])
fut1_dia = int(fut1_fila1['dia'])
fut1_hora = int(fut1_fila1['hora'])
fut1_minuto = int(fut1_fila1['minuto'])

# FUTURO_2: cogemos solo las velas de AAAAMMDD_HHMM de las que tenemos prediccion en FUT_1
datos_fut2 = datos_fut2[datos_fut2['anio'] == fut1_anio]
datos_fut2 = datos_fut2[datos_fut2['mes'] == fut1_mes]
datos_fut2 = datos_fut2[datos_fut2['dia'] == fut1_dia]
datos_fut2 = datos_fut2[datos_fut2['hora'] == fut1_hora]
datos_fut2 = datos_fut2[datos_fut2['minuto'] == fut1_minuto]

print("Velas útiles:")
print("FUT_1 --> " + str(datos_fut1.shape[0]) + " x " + str(datos_fut1.shape[1]))
print("FUT_2 --> " + str(datos_fut2.shape[0]) + " x " + str(datos_fut2.shape[1]))

if datos_fut1.shape[0] > datos_fut2.shape[0]:
    raise NameError("Futuro-2 (real) debe tener al menos todas las empresas de futuro-1 (predicho X días antes)!!")

datos_fut1_util = datos_fut1.drop('TARGET', axis=1)
datos_fut2_util = datos_fut2[['empresa', 'TARGET']].rename({'TARGET': 'TARGET_REAL'}, axis=1)

print("Cogemos futuro_1 (target PREDICHO con una PROBABILIDAD) y le añadimos las columnas de FUT_2 (target REAL):")
resultado = pd.merge(datos_fut1_util, datos_fut2_util, on='empresa', how='left')

print("Cogemos solo las columnas utiles (identificadores de fila). Las features ya las ha mirado el modelo; no debemos mirar nada a ojo...")
resultado = resultado[['empresa', 'mercado', 'anio', 'mes', 'dia', 'hora', 'minuto', 'TARGET_PREDICHO', 'TARGET_PREDICHO_PROB', 'TARGET_REAL']]

resultado = resultado.sort_values('TARGET_PREDICHO_PROB', ascending=False)

pathCsvResultado = dir_validacion + subgrupo + "_FUT1_y_FUT2_COMPLETO_PREDICCION.csv"
print("Guardando en: ")
resultado.to_csv(pathCsvResultado, index=False, sep='|')

print("--- Validador en Python: FIN ---")


