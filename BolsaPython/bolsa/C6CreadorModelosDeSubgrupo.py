import sys
import pandas as pd

print("---- CAPA 6 - Crear y almacenar varios modelos (para cada subgrupo) -------")
print("Tipo de problema: CLASIFICACION BINOMIAL (target es boolean)")

print ("PARAMETROS: ")
path_csv_entrada = sys.argv[0]
pathModelos = sys.argv[1]
print ("Path al CSV con los datos de entrada: %s" % path_csv_entrada)
print ("Directorio donde guardar los modelos: %s" % pathModelos)

print ("Cargar datos (CSV reducido)...")
data = pd.read_csv(path_csv_entrada)
print ("Mostramos las 5 primeras filas:")
data.head()
print ("DIVIDIR EL DATASET DE ENTRADA EN 3 PARTES: TRAIN (60%), TEST (20%), VALIDACION (20%)...")
ds_train, ds_test, ds_validacion = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

######## PENDIENTE




