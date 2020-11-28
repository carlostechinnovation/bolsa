import sys
import os
import pandas as pd
from IPython.display import HTML

##################### EXPLICACION #####################
# Entradas:
# - Carpeta Dropbox: contiene varios CSV (predicciones MANEJABLES para cada subgrupo, excluyendo el SG_0 que no es util)
# - Path al fichero CALIDAD.csv
# Salidas:
# - Fichero llamado 202XMMDD.html en la carpeta de Dropbox que agrupa  toda la info predicha para ese dia. Ademas, le añade columnas de CALIDAD.csv para poder entregarlo ordenado
###############################################################

print("--- InversionJuntarPrediccionesDeUnDia: INICIO ---")
entradaPathDirDropbox = sys.argv[1]
entradaPathCalidad = sys.argv[2]
targetPredichoProbUmbral = float(0.80)
print("entradaPathDirDropbox = " + entradaPathDirDropbox)
print("entradaPathCalidad = " + entradaPathCalidad)
print("targetPredichoProbUmbral = " + str(targetPredichoProbUmbral))

manejablesCsv = []

for item in os.listdir(entradaPathDirDropbox):
       if os.path.isfile(os.path.join(entradaPathDirDropbox, item)):
           if '.csv' in item and 'MANEJABLE' in item:
               manejablesCsv.append(item)

print("Excluimos el SG_0 porque sus predicciones no son utiles (el sistema predictivo del SG_0 es demasiado generalista)...")
manejablesCsv = [item for item in manejablesCsv if item.find('SG_0') == -1 ]
manejablesCsv.sort(reverse=True)  # ORDENAR (para coger el primer elemento)
primerFichero = manejablesCsv[0]
print("Primer fichero encontrado: " + primerFichero)
fecha = primerFichero.split("_")[0]
print("Fecha extraida del elemento mas reciente encontrado: " + fecha)
print("Solo cogemos los ficheros CSV de ese dia...")
manejablesCsv = [item for item in manejablesCsv if item.find(str(fecha)) != -1]
print("Numero de ficheros CSV de prediccion encontrados para el dia " + str(fecha) + ": " + str(len(manejablesCsv)) + " ficheros")

print("Leyendo fichero de CALIDAD de los subgrupos...")
calidadDatos = []
numFilasLeidas=0
if os.path.exists(entradaPathCalidad):
    calidadDatos = pd.read_csv(entradaPathCalidad, sep="|")
    numFilasLeidas = numFilasLeidas + calidadDatos.shape[0]
    calidadDatos.sort_values(by="calidadMediana", ascending=False, inplace=True)

print("Numero de filas leidas en CALIDAD.csv: " + str(numFilasLeidas))

numFilasLeidas = 0
juntos = pd.DataFrame()
for file in manejablesCsv:
    print("Procesando: " + file + " ...")
    prediccionesDeUnSubgrupo = pd.read_csv(entradaPathDirDropbox + file, sep="|")
    numFilasLeidas = numFilasLeidas + prediccionesDeUnSubgrupo.shape[0]
    prediccionesDeUnSubgrupo = prediccionesDeUnSubgrupo[["empresa", "mercado", "TARGET_PREDICHO_PROB", "NumAccionesPor1000dolares"]]
    idSubgrupo = file.split("_")[4]
    #print("idSubgrupo=" + idSubgrupo)

    # Dentro del DF de calidad, seleccionamos la de este subgrupo (si la hay)
    calidadDelSubgrupo = calidadDatos[calidadDatos['subgrupo'] == float(idSubgrupo)]
    if calidadDelSubgrupo.empty == False:
        calidadDelSubgrupoNfilas = calidadDelSubgrupo
        #Crear un dataframe de N filas (las mismas que predicciones) y con los mismos valores (los de calidad del subgrupo)
        calidadDelSubgrupoNfilas=calidadDelSubgrupoNfilas.loc[calidadDelSubgrupoNfilas.index.repeat(prediccionesDeUnSubgrupo.shape[0])].reset_index(drop=True)
        juntos=juntos.append(prediccionesDeUnSubgrupo.join(calidadDelSubgrupoNfilas))
    else:
        print("No tenemos datos de calidad para el idSubgrupo=" + idSubgrupo + ". Por tanto, dejamos vacias las columnas que describen la calidad de ese subgrupo.")
        prediccionesDeUnSubgrupo["subgrupo"] = idSubgrupo
        prediccionesDeUnSubgrupo["calidad"] = ""
        prediccionesDeUnSubgrupo["calidadMediana"] = ""
        prediccionesDeUnSubgrupo["calidadMediaStd"] = ""
        juntos=juntos.append(prediccionesDeUnSubgrupo)

juntos = juntos.sort_values(by=['calidadMediana', 'TARGET_PREDICHO_PROB'], ascending=False).reset_index(drop=True)

print("Numero de filas en fichero juntos: " + str(juntos.shape[0]))
print("Aplicando umbral en target_prob para coger solo las altas probabilidades...")
juntos = juntos[juntos['TARGET_PREDICHO_PROB'] >= targetPredichoProbUmbral]
print("Numero de filas en fichero juntos con alta probabilidad: " + str(juntos.shape[0]))
print("Reordenando columnas...")

juntos=juntos.reindex(['subgrupo', 'calidad', 'calidadMediana', 'calidadMediaStd', 'empresa', 'mercado', 'TARGET_PREDICHO_PROB', 'NumAccionesPor1000dolares'],axis=1)
pathSalida = entradaPathDirDropbox + str(fecha) + ".html"
print("Escribiendo en: " + pathSalida)
datosEnHtml = HTML(juntos.to_html(index=False, classes='table table-striped table-bordered table-hover table-condensed'))
text_file = open(pathSalida, "w")
text_file.write(datosEnHtml.data)
text_file.close()

print("--- InversionJuntarPrediccionesDeUnDia: FIN ---")
