###########################################################################
# LABORATORIO:
# Partiendo del fichero COMPLETO.CSV del SG0, vamos a ver si detectamos clusters útiles
###########################################################################
import pandas as pd
import glob

print("CLUSTERING - INICIO")
X = pd.DataFrame()
contador = 0
maximoDebug = 10000000
for pathCsvCompleto in glob.iglob('/bolsa/pasado/elaborados/*.csv'):
    if contador < maximoDebug:
        contador = contador + 1

        if contador%100==0:
            print(str(contador) + " --> " + pathCsvCompleto)

        # interpretamos que la primera columna es el indice (empresa) y leemos solo una fila porque son columnas estaticas (iguales en todas las velas)
        X1 = pd.read_csv(filepath_or_buffer=pathCsvCompleto, sep='|', nrows=1, index_col=0)

        # Columnas estaticas
        #columnasEstaticas = ['industria', 'Insider Own', 'Quick Ratio', 'Current Ratio', 'P/E', 'Dividend %',
        #                     'Employees', 'Short Ratio', 'geo', 'Short Float', 'Debt/Eq', 'LT Debt/Eq', 'P/S',
        #                     'EPS next Y', 'Recom', 'sector', 'Inst Own', 'Market Cap']
        columnasEstaticas = ['sector', 'geo', 'Insider Own', 'Debt/Eq']  # elijo solo 3 dimensiones (podria coger mas)
        X1 = X1[columnasEstaticas].replace('-', -1, regex=False)

        X = X.append(X1)

####################################################################3

# ----------- ENCODE CATEGORICAL FEAT---------------------------------
from sklearn.preprocessing import OrdinalEncoder

def codificarColumna(X, nombreCol):
    if nombreCol in X:
        ord_enc1 = OrdinalEncoder()
        X[nombreCol+"_code"] = ord_enc1.fit_transform(X[[nombreCol]])
        X.drop(nombreCol, axis=1, inplace=True)
        X.rename(columns={nombreCol+"_code": nombreCol})


codificarColumna(X, "industria")
codificarColumna(X, "sector")
codificarColumna(X, "geo")
# --------------------------------------------

X.drop(X.columns[1], axis=1, inplace=True) #borrar primera columna (es el ID)
X = X.dropna(axis=0, how='any')  # Borrar FILA si ALGUNO sus valores tienen NaN

from tabulate import tabulate
print(tabulate(X.head(3), headers='keys', tablefmt='psql'))

################# CLUSTERING ############################
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

bandwidth = estimate_bandwidth(X, quantile=0.3, n_samples=None, random_state=0, n_jobs=-1)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("Numero de clusters: %d" % n_clusters_)

#print("Etiquetas:")
labelsDF = pd.DataFrame(labels)
labelsDF.columns=['etiqueta']
labelsDF = pd.Series(data = labelsDF['etiqueta'].values, index = X.index.values).to_frame()
labelsDF.columns=['etiqueta']
labelsDF=labelsDF.sort_values(by=['etiqueta'])
#print(labelsDF)

estadisticas = labelsDF.groupby(['etiqueta'])
grupos = estadisticas.groups
gruposDF = pd.DataFrame.from_dict(grupos, orient="index").reset_index()
gruposDF=gruposDF.replace(to_replace = np.nan, value ='')  # valores None
gruposDF=gruposDF.rename(columns={"index": "cluster"})
# print(tabulate(gruposDF, headers='keys', tablefmt='psql', ))

pathSalida="/bolsa/pasado/empresas_clustering.csv"
pathSalidaHtml="/bolsa/pasado/empresas_clustering_web.html"
print("CLUSTERING - Escribiendo salida en: " + pathSalida)
print("CLUSTERING - Escribiendo salida HTML en: " + pathSalidaHtml)
print("CLUSTERING - El cluster gigante no debemos usarlo porque su análisis es demasiado laxo (se podria dividir aun mas)." + pathSalidaHtml)
labelsDF.to_csv(pathSalida, index=True, sep='|')
gruposDF.to_html(pathSalidaHtml, index=False)
# #############################################################################
print("CLUSTERING - FIN")