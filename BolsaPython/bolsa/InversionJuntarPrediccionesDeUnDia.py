import sys
import os
import pandas as pd
from IPython.display import HTML
import html, urllib3

##################### EXPLICACION #####################
# Entradas:
# - Carpeta Dropbox: contiene varios CSV (predicciones MANEJABLES para cada subgrupo, excluyendo el SG_0 que no es util)
# - Path al fichero CALIDAD.csv
# Salidas:
# - Fichero llamado 202XMMDD.html en la carpeta de Dropbox que agrupa toda la info predicha para ese dia. Ademas, le añade columnas de CALIDAD.csv para poder entregarlo ordenado
###############################################################

print("--- InversionJuntarPrediccionesDeUnDia: INICIO ---")
entradaPathDirDropbox = sys.argv[1]
entradaPathCalidad = sys.argv[2]
entradaPathDescripcionSubgrupos = sys.argv[3]
targetPredichoProbUmbral = float(0.91)
print("entradaPathDirDropbox = " + entradaPathDirDropbox)
print("entradaPathCalidad = " + entradaPathCalidad)
print("entradaPathDescripcionSubgrupos = " + entradaPathDescripcionSubgrupos)
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
if os.path.exists(entradaPathCalidad):
    calidadDatos = pd.read_csv(entradaPathCalidad, sep="|")
    calidadDatos = calidadDatos.drop("calidad", axis=1).drop("calidadMediaStd", axis=1)
    calidadDatos.sort_values(by="calidadMediana", ascending=False, inplace=True)

print("Numero de filas leidas en CALIDAD.csv: " + str(calidadDatos.shape[0]))


descripcionSubgrupos = []
if os.path.exists(entradaPathDescripcionSubgrupos):
    descripcionSubgrupos = pd.read_csv(entradaPathDescripcionSubgrupos, sep="|")

print("Numero de filas leidas en DESCRIPCION DE SUBGRUPOS: " + str(descripcionSubgrupos.shape[0]))


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
    descripcionDelSubgrupo = descripcionSubgrupos[descripcionSubgrupos['subgrupo'] == float(idSubgrupo)]
    if calidadDelSubgrupo.empty == False and descripcionDelSubgrupo.empty == False:

        #columnas de calidad del subgrupo
        calidadDelSubgrupoNfilas = calidadDelSubgrupo
        #Crear un dataframe de N filas (las mismas que predicciones) y con los mismos valores (los de calidad del subgrupo)
        calidadDelSubgrupoNfilas=calidadDelSubgrupoNfilas.loc[calidadDelSubgrupoNfilas.index.repeat(prediccionesDeUnSubgrupo.shape[0])].reset_index(drop=True)
        datosDeSubgrupo = prediccionesDeUnSubgrupo.join(calidadDelSubgrupoNfilas)

        # Columna "Descripcion de subgrupo"
        datosDeSubgrupo = pd.merge(datosDeSubgrupo, descripcionDelSubgrupo, how='outer', on='subgrupo')
        juntos=juntos.append(datosDeSubgrupo)
    else:
        print("No tenemos datos de CALIDAD o DESCRIPCION para el idSubgrupo=" + idSubgrupo + ". Por tanto, dejamos vacias las columnas que lo describen.")
        prediccionesDeUnSubgrupo["subgrupo"] = idSubgrupo
        prediccionesDeUnSubgrupo["calidadMediana"] = ""
        prediccionesDeUnSubgrupo["descripcion"] = ""

juntos = juntos.sort_values(by=['calidadMediana', 'TARGET_PREDICHO_PROB'], ascending=False).reset_index(drop=True)

print("Numero de filas en fichero juntos: " + str(juntos.shape[0]))
print("Aplicando umbral en target_prob para coger solo las altas probabilidades...")
juntos = juntos[juntos['TARGET_PREDICHO_PROB'] >= targetPredichoProbUmbral]
print("Numero de filas en fichero juntos con alta probabilidad: " + str(juntos.shape[0]))

# Ponemos link de FINVIZ en el nombre de la empresa, para poder hacer click
#meterLinkFinviz = lambda x: html.escape("""'<a href="www.google.es">" + x["empresa"]+"</a>'""")
#juntos["empresa"] = meterLinkFinviz(juntos)


print("Reordenando columnas...")
juntos = juntos.reindex(['subgrupo', 'descripcion', 'calidadMediana', 'empresa', 'mercado', 'TARGET_PREDICHO_PROB', 'NumAccionesPor1000dolares'],axis=1)
pathSalida = entradaPathDirDropbox + str(fecha) + ".html"
print("Escribiendo en: " + pathSalida)
datosEnHtml = HTML(juntos.to_html(index=False, classes='table table-striped table-bordered table-hover table-condensed'))
text_file = open(pathSalida, "w", encoding="utf-8")
text_file.writelines('<meta charset="UTF-8">\n')
text_file.write(datosEnHtml.data)
text_file.close()


#################################### FICHERO DE TODAS LAS EMPRESAS, SIN FILTRAR ###########################3
todasEmpresasYProbabsDF = pd.DataFrame(columns=["empresa"])
manejablesCsv.sort()
for file in manejablesCsv:
    print("Procesando: " + file + " ...")
    prediccionesDeUnSubgrupo = pd.read_csv(entradaPathDirDropbox + file, sep="|")
    prediccionesDeUnSubgrupo = prediccionesDeUnSubgrupo[["empresa", "TARGET_PREDICHO_PROB"]]
    idSubgrupo = file.split("_")[4]
    print("idSubgrupo=" + idSubgrupo)
    prediccionesDeUnSubgrupo.rename(columns={"TARGET_PREDICHO_PROB": "SG_"+idSubgrupo}, inplace=True)
    todasEmpresasYProbabsDF = pd.merge(todasEmpresasYProbabsDF, prediccionesDeUnSubgrupo, how='outer', left_on=['empresa'], right_on=['empresa'])


todasEmpresasYProbabsDF = todasEmpresasYProbabsDF.sort_values(by=['empresa'], ascending=True).reset_index(drop=True)  # Ordenar filas
todasEmpresasYProbabsDF = todasEmpresasYProbabsDF.fillna("")  # Sustituir NaN por cadena vacia para que quede bonito

todasEmpresasYProbabsDF_aux=todasEmpresasYProbabsDF.drop(['empresa'], axis=1, inplace=False)
import numpy as np
todasEmpresasYProbabsDF_aux['prob_media'] = todasEmpresasYProbabsDF_aux.replace('', np.nan).astype(float).mean(skipna=True, numeric_only=True, axis=1)
todasEmpresasYProbabsDF['prob_media']=todasEmpresasYProbabsDF_aux['prob_media']
todasEmpresasYProbabsDF.sort_values(by=['prob_media'], ascending=False, inplace=True)

pathSalida = entradaPathDirDropbox + str(fecha) + "_todas_las_empresas.html"
print("Escribiendo en: " + pathSalida)
datosEnHtml = HTML(todasEmpresasYProbabsDF.to_html(index=False, classes='table table-striped table-bordered table-hover table-condensed'))
text_file = open(pathSalida, "w", encoding="utf-8")
text_file.writelines('<meta charset="UTF-8">\n')
text_file.write(datosEnHtml.data)
text_file.close()
############################################################




print("--- InversionJuntarPrediccionesDeUnDia: FIN ---")

