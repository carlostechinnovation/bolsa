import os
import sys

import pandas as pd

print("--- Validador en Python: INICIO ---")
dir_validacion = sys.argv[1]
print("Directorio que contiene csv_fut1 y csv_fut2 es: " + dir_validacion)
ficheros = sorted(os.listdir(dir_validacion), key=len)
ficheros_csv = []
for fichero in ficheros:
    if fichero.endswith("csv"):
        ficheros_csv.append(fichero)


ficherosDF = pd.DataFrame(ficheros_csv, columns=["ficheronombre"])
ficherosDF['antiguedad'] = ficherosDF.apply(lambda fila: fila.ficheronombre.split("_")[0], axis=1)  # nueva columna
ficherosDF['subgrupo'] = ficherosDF.apply(lambda fila: fila.ficheronombre.split("_")[2], axis=1)  # nueva columna

#ordenar por subgrupo y antiguedad, ascendente
ficherosDF = ficherosDF.sort_values(['subgrupo', 'antiguedad'],ascending=[True, True])

if len(ficheros_csv) % 2 !=0:
    raise NameError("La carpeta de validacion deberia tener un numero PAR de ficheros CSV (fut1 y fut2) para poder compararlos: " + dir_validacion)


listaAntiguedades = ficherosDF['antiguedad'].unique().tolist()  # Lista única de antiguedades
listaSubgrupos = ficherosDF['subgrupo'].unique().tolist()  # Lista única de subgrupos

listaSubgrupos = list(map(int, listaSubgrupos))  # Convertir lista de string a int
listaSubgrupos.sort()  # ordenar ascendente


for subgrupo in listaSubgrupos:
    print("Procesando ficheros CSV de subgrupo: " + str(subgrupo))
    csv_fut1 = dir_validacion + listaAntiguedades[1] + "_SG_" + str(subgrupo) + "_COMPLETO_PREDICCION.csv"  # Futuro 1 es el más antiguo
    csv_fut2 = dir_validacion + listaAntiguedades[0] + "_SG_" + str(subgrupo) + "_COMPLETO_PREDICCION.csv"  # Futuro 1 es el más reciente

    datos_fut1 = pd.read_csv(filepath_or_buffer=csv_fut1, sep='|')
    datos_fut2 = pd.read_csv(filepath_or_buffer=csv_fut2, sep='|')

    # FUTURO_1: cogemos las predicciones target=1
    filas_con_futuro_predicho = datos_fut1['TARGET_PREDICHO'] == 1
    datos_fut1 = datos_fut1[filas_con_futuro_predicho]

    # Numero de filas con TARGET_PREDICHO=1
    numPredichosPositivos = datos_fut1.shape[0]

    if numPredichosPositivos == 0:
        print("El subgrupo " + str(subgrupo) + " no tiene ningun caso con TARGET_PREDICHO=1. Por ello, no generamos fichero...")
    else:
        asd=0

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

        pathCsvResultado = dir_validacion + str(subgrupo) + "_FUT1_y_FUT2_COMPLETO_PREDICCION.csv"
        print("Guardando en: " + pathCsvResultado)
        resultado.to_csv(pathCsvResultado, index=False, sep='|')


print("--- Validador en Python: FIN ---")


