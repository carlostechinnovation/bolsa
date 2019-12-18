import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

print("---- CAPA 6 - Crear y almacenar varios modelos (para cada subgrupo) -------")
print("Tipo de problema: CLASIFICACION BINOMIAL (target es boolean)")

print ("PARAMETROS: ")
dir_csvs_entrada = sys.argv[1]
pathModelos = sys.argv[2]
print ("path_csv_entrada: %s" % dir_csvs_entrada)
print ("pathModelos: %s" % pathModelos)


print("Recorremos los CSVs que hay en el DIRECTORIO...")
for entry in os.listdir(dir_csvs_entrada):
  path_absoluto_fichero = os.path.join(dir_csvs_entrada, entry)

  if (entry.endswith('.csv') and os.path.isfile(path_absoluto_fichero) and os.stat(path_absoluto_fichero).st_size > 0 ):
    print("-------------------------------------------------------------------------------------------------")
    id_subgrupo = Path(entry).stem
    print("id_subgrupo=" + id_subgrupo)
    pathEntrada = os.path.abspath(entry)
    print ("Cargar datos (CSV reducido)...")
    inputFeaturesyTarget = pd.read_csv(path_absoluto_fichero)

    print ("DIVIDIR EL DATASET DE ENTRADA EN 3 PARTES: TRAIN (60%), TEST (20%), VALIDACION (20%)...")
    ds_train, ds_test, ds_validacion = np.split(inputFeaturesyTarget.sample(frac=1), [int(.6 * len(inputFeaturesyTarget)), int(.8 * len(inputFeaturesyTarget))])
    print ("TRAIN --> " + str(ds_train.shape[0]) + " x " + str(ds_train.shape[1]))
    print ("TEST --> " + str(ds_test.shape[0]) + " x " + str(ds_test.shape[1]))
    print ("VALIDACION --> " + str(ds_validacion.shape[0]) + " x " + str(ds_validacion.shape[1]))


################## MAIN ########################################
print("PENDIENTE...")



############################################################
print("------------ FIN ----------------")


