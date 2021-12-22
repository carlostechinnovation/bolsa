##########################################################################################
# LABORATORIO:
# Partiendo del fichero COMPLETO.CSV del SG0, vamos a intentar detectar clusters útiles
##########################################################################################
import pandas as pd
import glob
import pickle
import sys
import numpy as np

print("CLUSTERING - INICIO")

print("PARAMETROS: ")
modo = sys.argv[1]  # pasado o futuro

################### Preparacion de la entrada #######################################################################
X = pd.DataFrame()
contador = 0
maximoDebug = 1000000
for pathCsvCompleto in glob.iglob("/bolsa/" + modo + "/elaborados/*.csv"):
    if contador < maximoDebug:
        contador = contador + 1

        if contador % 100 == 0:
            print(str(contador) + " --> " + pathCsvCompleto)

        # interpretamos que la primera columna es el indice (empresa) y leemos solo una fila porque son columnas estaticas (iguales en todas las velas)
        X1 = pd.read_csv(filepath_or_buffer=pathCsvCompleto, sep='|', nrows=1, index_col=0)

        # Columnas estaticas
        #columnasEstaticas = ['industria', 'Insider Own', 'Quick Ratio', 'Current Ratio', 'P/E', 'Dividend %',
        #                     'Employees', 'Short Ratio', 'geo', 'Short Float', 'Debt/Eq', 'LT Debt/Eq', 'P/S',
        #                     'EPS next Y', 'Recom', 'sector', 'Inst Own', 'Market Cap']
        columnasEstaticas = ['sector', 'geo', 'Insider Own', 'Debt/Eq']  # elijo solo unas dimensiones (podria coger mas)
        X1 = X1[columnasEstaticas].replace('-', -1, regex=False)
        X = X.append(X1)
        X1 = ""


# ----------- ENCODE CATEGORICAL FEAT---------------------------------
from sklearn.preprocessing import OrdinalEncoder

def codificarColumna(X, nombreCol, pathEncoderSalida):
    if nombreCol in X:
        if modo == "pasado":
            ord_enc1 = OrdinalEncoder()
            X[nombreCol+"_code"] = ord_enc1.fit_transform(X[[nombreCol]])
            pickle.dump(ord_enc1, open(pathEncoderSalida, 'wb'))
        elif modo == "futuro":
            ord_enc1 = pickle.load(open(pathEncoderSalida, 'rb'))
            valoresEntrenados = np.array(ord_enc1.categories_)
            valoresEntrada = X[nombreCol].unique()
            valoresNoSoportados = np.setdiff1d(valoresEntrada, valoresEntrenados)

            print("Para la columna " + nombreCol + " estos valores no aparecieron al entrenar el modelo de clustering, así que descartaremos las filas del FUTURO en las que aparecen: " + np.array2string(valoresNoSoportados, separator=',') )
            X = X[~X[nombreCol].isin(valoresNoSoportados)]
            print("nombreCol: " + nombreCol)

            pd.options.mode.chained_assignment = None  # protege de un warning en la linea siguiente
            X[nombreCol + "_code"] = ord_enc1.transform(X[[nombreCol]])
        else:
            raise Exception("ERROR 001 - No esperado. Saliendo...")

        # Comun a pasado y futuro
        X.drop(nombreCol, axis=1, inplace=True)
        X.rename(columns={nombreCol + "_code": nombreCol})

    return X


X = codificarColumna(X, "industria", "/bolsa/" + modo + "/clustering_encoder_industria.dat")
X = codificarColumna(X, "sector", "/bolsa/" + modo + "/clustering_encoder_sector.dat")
X = codificarColumna(X, "geo", "/bolsa/" + modo + "/clustering_encoder_geo.dat")
# --------------------------------------------

X.drop(X.columns[1], axis=1, inplace=True)  # borrar primera columna (es el ID)
X = X.dropna(axis=0, how='any')  # Borrar FILA si ALGUNO sus valores tienen NaN

from tabulate import tabulate
print(tabulate(X.head(3), headers='keys', tablefmt='psql'))

################# MODELO DE CLUSTERING: entrenar+usar (pasado) o usar (futuro) ############################
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

pathModelo = "/bolsa/pasado/clustering_modelo.dat"  # Siempre PASADO (nunca futuro)
labels = []
if modo == "pasado":
    bandwidth = estimate_bandwidth(X, quantile=0.3, n_samples=None, random_state=0, n_jobs=-1)
    modelo = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    modelo.fit(X)
    labels = modelo.labels_
    pickle.dump(modelo, open(pathModelo, 'wb'))
    cluster_centers = modelo.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("Numero de clusters (pasado): %d" % n_clusters_)
elif modo == "futuro":
    print("Aplicamos modelo  de clustering ya entrenado para asignar clusters ya definados a las muestras del futuro")
    modelo = pickle.load(open(pathModelo, 'rb'))
    labels = modelo.predict(X)
else:
    raise Exception("ERROR 002 - No esperado. Saliendo...")

print("Etiquetas (IDs de cluster)...")
labelsDF = pd.DataFrame(labels)
labelsDF.columns = ['etiqueta']
labelsDF = pd.Series(data=labelsDF['etiqueta'].values, index=X.index.values).to_frame()
labelsDF.columns = ['etiqueta']
labelsDF = labelsDF.sort_values(by=['etiqueta'])
# print(labelsDF)

estadisticas = labelsDF.groupby(['etiqueta'])
grupos = estadisticas.groups
gruposDF = pd.DataFrame.from_dict(grupos, orient="index").reset_index()
gruposDF = gruposDF.replace(to_replace = np.nan, value ='')  # valores None
gruposDF = gruposDF.rename(columns={"index": "cluster"})
# print(tabulate(gruposDF, headers='keys', tablefmt='psql', ))

pathSalida = "/bolsa/" + modo + "/empresas_clustering.csv"
print("CLUSTERING - Escribiendo salida en: " + pathSalida)
labelsDF.to_csv(pathSalida, index=True, sep='|')

pathSalidaHtml = "/bolsa/" + modo + "/empresas_clustering_web.html"
print("CLUSTERING - Escribiendo salida HTML en: " + pathSalidaHtml)
print("CLUSTERING - El cluster gigante no debemos usarlo porque su análisis es demasiado laxo (se podria dividir aun mas).")
gruposDF.to_html(pathSalidaHtml, index=False)
# #############################################################################

print("CLUSTERING - FIN")

