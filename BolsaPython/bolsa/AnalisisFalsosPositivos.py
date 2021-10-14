import os
import sys
import pandas as pd
import glob
from tabulate import tabulate

##################### EXPLICACION #####################
# Tras haber ejecutado en modo PASADO, se han generado ficheros con velas FP, falsos positivos (predicho True, pero la realidad fue False).
# Queremos analizar por qué los modelos se han equivocado y si hay
# alguna característica común que podamos modelar con una NUEVA FEATURE ELABORADA.
###############################################################

print("=========== ANALISIS DE FALSOS POSITIVOS ==============")
dirPasadoSubgrupos = "/bolsa/pasado/subgrupos/"
print("dirPasadoSubgrupos = " + dirPasadoSubgrupos)
dirLogs = "/bolsa/logs/"
print("dirLogs = " + dirLogs)

##### RUTAS #####
pathFilesFp = []
ficherosFPContador = 0
for filename in glob.iglob(dirPasadoSubgrupos + '**/*.csv', recursive=True):
    if "falsospositivos" in filename:  # MODO: TEST y VALIDACION
        pathFilesFp.append(filename)
        ficherosFPContador += 1

##### BRUTO FALSOS POSITIVOS #####
velasFPbrutas = pd.DataFrame([], columns=["subgrupo", "modo", "dato"])  # MODO: TEST y VALIDACION
velasFPbrutas_temp = pd.DataFrame([], columns=["subgrupo", "modo", "dato"])  # MODO: TEST y VALIDACION
filasFPBrutas = 0
for ficheroFP in pathFilesFp:
    if os.path.exists(ficheroFP):
        velasFPbrutas_temp = pd.read_csv(ficheroFP, sep="|")
        if velasFPbrutas_temp.size > 0:
            velasFPbrutas_temp.columns = ["subgrupo", "modo", "dato"]
            filasFPBrutas += velasFPbrutas_temp.shape[0]
            if velasFPbrutas.empty:
                velasFPbrutas = velasFPbrutas_temp
            else:
                velasFPbrutas = velasFPbrutas.append(velasFPbrutas_temp)

##### LIMPIO FALSOS POSITIVOS #####
velasFPbrutas[['empresa', 'anio', 'mes', 'dia']] = velasFPbrutas['dato'].str.split('_', 4, expand=True)
velasFP = velasFPbrutas.drop('dato', axis=1)
velasFP['fecha'] = pd.to_datetime(velasFP.rename(columns={'anio': 'year', 'mes': 'month', 'dia': 'day'}, inplace=False)[['year', 'month', 'day']])
velasFP['diaensemana'] = pd.DatetimeIndex(velasFP['fecha']).day_name()
# print("Ejemplos:"); print(velasFP.head())

########## NUMERO DE PREDICCIONES CALCULADAS (TEST + VALIDACION) EN CADA SUBGRUPO #############
pathFilesPrediccionesTestYValid = []
for filename in glob.iglob(dirPasadoSubgrupos + '**/intermedio.csv.test_t_predicho.csv', recursive=True):
    pathFilesPrediccionesTestYValid.append(filename)
for filename in glob.iglob(dirPasadoSubgrupos + '**/intermedio.csv.validac_t_predicho.csv', recursive=True):
    pathFilesPrediccionesTestYValid.append(filename)

prediccionesTV = pd.DataFrame([], columns=["subgrupo", "numeroPredicciones"])
prediccionesTV_temp = pd.DataFrame([], columns=["indice", "targetpredicho"])
for ficheroPredTV in pathFilesPrediccionesTestYValid:
    sgExtraido = ficheroPredTV.split("/")[4]
    if os.path.exists(ficheroPredTV):
        prediccionesTV_temp = pd.read_csv(ficheroPredTV, sep="|")
        if prediccionesTV_temp.size > 0:
            prediccionesTV_temp.columns = ["indice", "targetpredicho"]
            numFilasTmp = prediccionesTV_temp.shape[0]
            nuevafila = {'subgrupo': sgExtraido, 'numeroPredicciones': numFilasTmp}
            prediccionesTV = prediccionesTV.append(nuevafila, ignore_index=True)

# Sumando los subtotales de Test y validacion
prediccionesTV = prediccionesTV.groupby('subgrupo').sum()


##### ANALISIS ########
print("Ficheros CSV leidos (test y validation): " + str(ficherosFPContador))
print("Velas leidas (falsos positivos): " + str(velasFP.shape[0]))
print("Numero de empresas, anios, meses, etc distintos analizados: ")
print(tabulate(velasFP.nunique().to_frame().transpose(), headers='keys', tablefmt='psql'))

print("Top EMPRESAS con MAS falsos positivos (para no invertir en ellas):")
data1 = velasFP.groupby('empresa')['dia'].count().to_frame().sort_values(by=['dia'], ascending=False)
data1.rename(columns={'dia': 'numvelasfp'}, inplace=True)
print("FALSOSPOSITIVOS - EMPRESAS - Path: " + dirLogs + "falsospositivos_empresas.csv")
data1.to_csv(dirLogs + "falsospositivos_empresas.csv", index=True, sep='|', float_format='%.4f')
print(tabulate(data1.head(20).transpose(), headers='keys', tablefmt='psql'))

print("Top MESES con MENOS falsos positivos:")
data1 = velasFP.groupby('mes')['dia'].count().to_frame().sort_values(by=['dia'], ascending=True)
data1.rename(columns={'dia': 'numvelasfp'}, inplace=True)
print("FALSOSPOSITIVOS - MESES - Path: " + dirLogs + "falsospositivos_meses.csv")
data1.to_csv(dirLogs + "falsospositivos_meses.csv", index=True, sep='|', float_format='%.4f')
print(tabulate(data1.head(20).transpose(), headers='keys', tablefmt='psql'))

print("Top DIA DE LA SEMANA con MENOS falsos positivos:")
data1 = velasFP.groupby('diaensemana')['dia'].count().to_frame().sort_values(by=['dia'], ascending=True)
data1.rename(columns={'diaensemana': 'numvelasfp'}, inplace=True)
print("FALSOSPOSITIVOS - DIAENSEMANA - Path: " + dirLogs + "falsospositivos_diaensemana.csv")
data1.to_csv(dirLogs + "falsospositivos_diaensemana.csv", index=True, sep='|', float_format='%.4f')
print(tabulate(data1.head(20).transpose(), headers='keys', tablefmt='psql'))


print("Top SUBGRUPOS con MENOS falsos positivos (relativo):")
data1 = velasFP.groupby('subgrupo')['dia'].count().to_frame().sort_values(by=['dia'], ascending=True)
data1 = data1.rename(columns={"dia":"numvelasfp"})
# print(tabulate(data1.transpose(), headers='keys', tablefmt='psql'))
data2 = prediccionesTV.sort_values(by=['subgrupo'], ascending=True)
# print(tabulate(data2.transpose(), headers='keys', tablefmt='psql'))
data3 = data1.merge(data2, how='inner', on='subgrupo')
data3['ratioFalsosPositivos'] = 100 * data3['numvelasfp'] / data3['numeroPredicciones']
data3 = data3.sort_values(by=['ratioFalsosPositivos'], ascending=True).round(1)
print("FALSOSPOSITIVOS - SUBGRUPOS - Path: " + dirLogs + "falsospositivos_subgrupos.csv")
data3.to_csv(dirLogs + "falsospositivos_subgrupos.csv", index=True, sep='|', float_format='%.4f')
print(tabulate(data3.transpose(), headers='keys', tablefmt='psql'))

print("Esto podria estar correlado con el OVERFITTING y la metrica ESPERADA de cada SUBGRUPO.")
print("============================================")

