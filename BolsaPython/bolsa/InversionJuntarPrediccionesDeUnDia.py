import os
import sys
import numpy as np
import pandas as pd
from IPython.display import HTML

##################### EXPLICACION #####################
# Entradas:
# - Carpeta Dropbox: contiene varios CSV (predicciones MANEJABLES para cada subgrupo, excluyendo el SG_0 que no es util)
# - Path al fichero CALIDAD.csv
# - Path al CSV de falsos positivos segun empresas
# - Path al fichero de FALSOS POSITIVOS agrupado por empresas
# Salidas:
# - Fichero llamado 202XMMDD.html en la carpeta de Dropbox que agrupa toda la info predicha para ese dia. Ademas, le añade columnas de CALIDAD.csv para poder entregarlo ordenado
###############################################################

print("--- InversionJuntarPrediccionesDeUnDia: INICIO ---")
# /home/carloslinux/Dropbox/BOLSA_PREDICTOR/ /home/carloslinux/Dropbox/BOLSA_PREDICTOR/ANALISIS/CALIDAD.csv /home/carloslinux/Desktop/GIT_BOLSA/bolsa/BolsaJava/src/main/resources/Bolsa_Subgrupos_Descripcion.txt /home/carloslinux/Desktop/GIT_BOLSA/bolsa/BolsaJava/realimentacion/falsospositivos_empresas.csv
entradaPathDirDropbox = sys.argv[1]
entradaPathCalidad = sys.argv[2]
entradaPathDescripcionSubgrupos = sys.argv[3]
entradaFPempresas = sys.argv[4]
targetPredichoProbUmbral = float(0.80)
print("entradaPathDirDropbox = " + entradaPathDirDropbox)
print("entradaPathCalidad = " + entradaPathCalidad)
print("entradaPathDescripcionSubgrupos = " + entradaPathDescripcionSubgrupos)
print("entradaFPempresas = " + entradaFPempresas)
print("targetPredichoProbUmbral = " + str(targetPredichoProbUmbral))

manejablesCsv = []

for item in os.listdir(entradaPathDirDropbox):
    if os.path.isfile(os.path.join(entradaPathDirDropbox, item)):
        if '.csv' in item and 'MANEJABLE' in item:
            manejablesCsv.append(item)

print(
    "Excluimos el SG_0 porque sus predicciones no son utiles (el sistema predictivo del SG_0 es demasiado generalista)...")
manejablesCsv = [item for item in manejablesCsv if item.find('SG_0') == -1]
manejablesCsv.sort(reverse=True)  # ORDENAR (para coger el primer elemento)
primerFichero = manejablesCsv[0]
print("Primer fichero encontrado: " + primerFichero)
fecha = primerFichero.split("_")[0]
print("Fecha extraida del elemento mas reciente encontrado: " + fecha)
print("Solo cogemos los ficheros CSV de ese dia...")
manejablesCsv = [item for item in manejablesCsv if item.find(str(fecha)) != -1]
print("Numero de ficheros CSV de prediccion encontrados para el dia " + str(fecha) + ": " + str(
    len(manejablesCsv)) + " ficheros")

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

print("Leyendo fichero de FALSOS POSITIVOS de las empresas: " + entradaFPempresas)
falsosPositivosEmpresas = []
if os.path.exists(entradaFPempresas):
    falsosPositivosEmpresas = pd.read_csv(entradaFPempresas, sep="|")
    falsosPositivosEmpresas = falsosPositivosEmpresas[["empresa", "ratioFalsosPositivos"]]
    falsosPositivosEmpresas.sort_values(by="empresa", ascending=True, inplace=True)

numFilasLeidas = 0
juntos = pd.DataFrame()
for file in manejablesCsv:
    print("Procesando: " + file + " ...")
    prediccionesDeUnSubgrupo = pd.read_csv(entradaPathDirDropbox + file, sep="|")
    numFilasLeidas = numFilasLeidas + prediccionesDeUnSubgrupo.shape[0]
    prediccionesDeUnSubgrupo = prediccionesDeUnSubgrupo[
        ["empresa", "mercado", "TARGET_PREDICHO_PROB", "NumAccionesPor1000dolares"]]
    idSubgrupo = file.split("_")[4]
    # print("idSubgrupo=" + idSubgrupo)

    # Dentro del DF de calidad, seleccionamos la de este subgrupo (si la hay)
    calidadDelSubgrupo = calidadDatos[calidadDatos['subgrupo'] == float(idSubgrupo)]
    descripcionDelSubgrupo = descripcionSubgrupos[descripcionSubgrupos['subgrupo'] == float(idSubgrupo)]
    if calidadDelSubgrupo.empty == False and descripcionDelSubgrupo.empty == False:

        # columnas de calidad del subgrupo
        calidadDelSubgrupoNfilas = calidadDelSubgrupo
        # Crear un dataframe de N filas (las mismas que predicciones) y con los mismos valores (los de calidad del subgrupo)
        calidadDelSubgrupoNfilas = calidadDelSubgrupoNfilas.loc[
            calidadDelSubgrupoNfilas.index.repeat(prediccionesDeUnSubgrupo.shape[0])].reset_index(drop=True)
        datosDeSubgrupo = prediccionesDeUnSubgrupo.join(calidadDelSubgrupoNfilas)

        # Columna "Descripcion de subgrupo"
        datosDeSubgrupo = pd.merge(datosDeSubgrupo, descripcionDelSubgrupo, how='outer', on='subgrupo')
        juntos = juntos.append(datosDeSubgrupo)
    else:
        print(
            "No tenemos datos de CALIDAD o DESCRIPCION para el idSubgrupo=" + idSubgrupo + ". Por tanto, dejamos vacias las columnas que lo describen.")
        prediccionesDeUnSubgrupo["subgrupo"] = idSubgrupo
        prediccionesDeUnSubgrupo["calidadMediana"] = ""
        prediccionesDeUnSubgrupo["descripcion"] = ""

juntos = juntos.sort_values(by=['calidadMediana', 'TARGET_PREDICHO_PROB'], ascending=False).reset_index(drop=True)

print("Numero de filas en fichero juntos: " + str(juntos.shape[0]))
juntos['calidadMediana'] = juntos['calidadMediana'].round(decimals=2)  # redondear decimales
juntos['TARGET_PREDICHO_PROB'] = juntos['TARGET_PREDICHO_PROB'].round(decimals=2)  # redondear decimales

print("Aplicando umbral en target_prob para coger solo las altas probabilidades...")
juntos = juntos[juntos['TARGET_PREDICHO_PROB'] >= targetPredichoProbUmbral]
print("Numero de filas en fichero juntos con alta probabilidad: " + str(juntos.shape[0]))

# Ponemos link de FINVIZ en el nombre de la empresa, para poder hacer click
# meterLinkFinviz = lambda x: html.escape("""'<a href="www.google.es">" + x["empresa"]+"</a>'""")
# juntos["empresa"] = meterLinkFinviz(juntos)


print("Reordenando columnas...")
juntos = juntos.reindex(['subgrupo', 'descripcion', 'calidadMediana', 'empresa', 'mercado', 'TARGET_PREDICHO_PROB',
                         'NumAccionesPor1000dolares'], axis=1)

juntos.rename(columns={"TARGET_PREDICHO_PROB": "PROBA"}, inplace=True)  # renombrar columna

pathSalida = entradaPathDirDropbox + str(fecha) + ".html"
print("Escribiendo en: " + pathSalida)
datosEnHtml = HTML(
    juntos.to_html(index=False, classes='table table-striped table-bordered table-hover table-condensed'))
contenidoHtml = datosEnHtml.data
contenidoHtml = contenidoHtml.replace("<th>empresa</th>",
                                      "<th style=\"background-color: yellow;\">empresa</th>")  # Colores
contenidoHtml = contenidoHtml.replace("<th>PROBA</th>",
                                      "<th style=\"background-color: yellow;\">prob_media</th>")  # Colores
text_file = open(pathSalida, "w", encoding="utf-8")
text_file.writelines('<meta charset="UTF-8">\n')
text_file.write(contenidoHtml)
text_file.close()

#################################### FICHERO DE TODAS LAS EMPRESAS, SIN FILTRAR ###########################3
todasEmpresasYProbabsDF = pd.DataFrame(columns=["empresa"])
manejablesCsv.sort()
for file in manejablesCsv:
    print("Procesando: " + file + " ...")
    prediccionesDeUnSubgrupo = pd.read_csv(entradaPathDirDropbox + file, sep="|")
    prediccionesDeUnSubgrupo = prediccionesDeUnSubgrupo[["empresa", "TARGET_PREDICHO_PROB"]]
    prediccionesDeUnSubgrupo['TARGET_PREDICHO_PROB'] = prediccionesDeUnSubgrupo['TARGET_PREDICHO_PROB'].round(
        decimals=2)  # redondear decimales
    idSubgrupo = file.split("_")[4]
    print("idSubgrupo=" + idSubgrupo)
    prediccionesDeUnSubgrupo.rename(columns={"TARGET_PREDICHO_PROB": "SG_" + idSubgrupo}, inplace=True)
    todasEmpresasYProbabsDF = pd.merge(todasEmpresasYProbabsDF, prediccionesDeUnSubgrupo, how='outer',
                                       left_on=['empresa'], right_on=['empresa'])

todasEmpresasYProbabsDF = todasEmpresasYProbabsDF.sort_values(by=['empresa'], ascending=True).reset_index(
    drop=True)  # Ordenar filas
todasEmpresasYProbabsDF = todasEmpresasYProbabsDF.fillna("")  # Sustituir NaN por cadena vacia para que quede bonito
todasEmpresasYProbabsDF_aux = todasEmpresasYProbabsDF.drop(['empresa'], axis=1, inplace=False)

######################################################################################
print("REORDENAR COLUMNAS ALFABETICAMENTE...")
todasEmpresasYProbabsDF_columnasSorted = todasEmpresasYProbabsDF_aux\
    .reindex(sorted(todasEmpresasYProbabsDF_aux, key=lambda x: float(x[3:])), axis=1)
todasEmpresasYProbabsDF_columnasSorted = pd.DataFrame(todasEmpresasYProbabsDF["empresa"]).join(todasEmpresasYProbabsDF_columnasSorted)
todasEmpresasYProbabsDF = todasEmpresasYProbabsDF_columnasSorted
######################################################################################

todasEmpresasYProbabsDF_aux['prob_media'] = todasEmpresasYProbabsDF_aux.replace('', np.nan).astype(float).mean(
    skipna=True, numeric_only=True, axis=1)
todasEmpresasYProbabsDF['prob_media'] = todasEmpresasYProbabsDF_aux['prob_media']
todasEmpresasYProbabsDF_aux['num_sg'] = todasEmpresasYProbabsDF_aux.drop('prob_media', axis=1).replace('', np.nan).astype(float).count(axis=1)
todasEmpresasYProbabsDF['num_sg'] = todasEmpresasYProbabsDF_aux['num_sg']
todasEmpresasYProbabsDF.sort_values(by=['prob_media'], ascending=False, inplace=True)

# Operaciones de insiders (tambien son buen indicador, pero lo ponemos para evitar entrar a mirar en Finviz manualmente)
print("Se incluyen columnas de operaciones con insiders...")
if os.path.isfile("/bolsa/futuro/subgrupos/SG_46/COMPLETO.csv"):
    entradaSG46df = pd.read_csv("/bolsa/futuro/subgrupos/SG_46/COMPLETO.csv", sep="|")
    entradaSG46df = entradaSG46df[entradaSG46df['antiguedad'] == 0]
    entradaSG46df = entradaSG46df[
        ["empresa", "flagOperacionesInsiderUltimos90dias", "flagOperacionesInsiderUltimos30dias",
         "flagOperacionesInsiderUltimos15dias", "flagOperacionesInsiderUltimos5dias"]] \
        .rename(columns={"flagOperacionesInsiderUltimos90dias": "I90D", "flagOperacionesInsiderUltimos30dias": "I30D",
                         "flagOperacionesInsiderUltimos15dias": "I15D", "flagOperacionesInsiderUltimos5dias": "I5D"})
else:
    entradaSG46df = pd.DataFrame([], columns=['empresa'])  # default: vacio

todasEmpresasYProbabsDFconInsiders = pd.merge(todasEmpresasYProbabsDF, entradaSG46df, how="left", on="empresa")
todasEmpresasYProbabsDFconInsiders = todasEmpresasYProbabsDFconInsiders.fillna(
    "")  # Sustituir NaN por cadena vacia para que quede bonito

todasEmpresasConFP = pd.merge(todasEmpresasYProbabsDFconInsiders, falsosPositivosEmpresas, how="left", on="empresa")
todasEmpresasConFP = todasEmpresasConFP.fillna("")  # Sustituir NaN por cadena vacia para que quede bonito
todasEmpresasConFP.rename(columns={"ratioFalsosPositivos": "ratioFalsosPositivos pasado (%)"},
                          inplace=True)  # renombrar columna

# Escritura a fichero HTML
pathSalida = entradaPathDirDropbox + str(fecha) + "_todas_las_empresas.html"
print("Escribiendo en: " + pathSalida)
datosEnHtml = HTML(todasEmpresasConFP.to_html(index=False,
                                              classes='table table-striped table-bordered table-hover table-condensed'))
contenidoHtml = datosEnHtml.data
contenidoHtml = contenidoHtml.replace("<td></td>",
                                      "<td style=\"background-color: lightsteelblue;\"></td>")  # Colores de celda vacia
contenidoHtml = contenidoHtml.replace("<th>empresa</th>",
                                      "<th style=\"background-color: yellow;\">empresa</th>")  # Colores
contenidoHtml = contenidoHtml.replace("<th>prob_media</th>",
                                      "<th style=\"background-color: yellow;\">prob_media</th>")  # Colores
contenidoHtml = contenidoHtml.replace("<td>-1.0</td>",
                                      "<td style=\"background-color: lightcoral;\">-1</td>")  # Colores de las Ventas
contenidoHtml = contenidoHtml.replace("<td>1.0</td>",
                                      "<td style=\"background-color: greenyellow;\">1</td>")  # Colores de las Compras
text_file = open(pathSalida, "w", encoding="utf-8")
text_file.writelines('<meta charset="UTF-8">\n')
text_file.write(contenidoHtml)
text_file.close()
############################################################

print("--- InversionJuntarPrediccionesDeUnDia: FIN ---")
