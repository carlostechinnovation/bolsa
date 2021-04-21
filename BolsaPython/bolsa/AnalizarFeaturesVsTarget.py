import sys
import os
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

print("AnalizarFeaturesVsTarget - INICIO")

#Evito el subgrupo 0 porque su CSV es tan gigante que tarda mucho
for idSubgrupo in range(1, 100):
    print("Analizando las features del SUBGRUPO: " +str(idSubgrupo))

    pathEntrada = "/bolsa/pasado/subgrupos/SG_" + str(idSubgrupo) + "/COMPLETO.csv"
    path_dir_img = "/bolsa/pasado/subgrupos/SG_" + str(idSubgrupo) + "/img/"
    variableAnalizada = "CURTOSIS_20_OPENHIGH"
    print("pathEntrada = %s" % pathEntrada)
    sns.set_theme()

    if os.path.exists(pathEntrada) and os.path.exists(pathEntrada):
        datosEntradaDF = pd.read_csv(filepath_or_buffer=pathEntrada, sep='|');

        for columna in datosEntradaDF:
            sns.jointplot(x=datosEntradaDF[columna], y=datosEntradaDF['TARGET']);
            path_dibujo = path_dir_img + columna + "_VS_TARGET" + ".png";
            print("Guardando:" + path_dibujo)
            plt.savefig(path_dibujo, bbox_inches='tight');
            plt.clf();
            plt.cla();
            plt.close()  # Limpiando dibujo

print("AnalizarFeaturesVsTarget - INICIO")

