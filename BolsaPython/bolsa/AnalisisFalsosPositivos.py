import glob
import os
import shutil
import sys

import pandas as pd
from tabulate import tabulate

##################### EXPLICACION #####################
# Tras haber ejecutado en modo PASADO, se han generado ficheros con velas FP, falsos positivos (predicho True, pero la realidad fue False).
# Queremos analizar por qué los modelos se han equivocado y si hay
# alguna característica común que podamos modelar con una NUEVA FEATURE ELABORADA.
###############################################################

print("=========== ANALISIS DE FALSOS POSITIVOS ==============")

##################################################################################################
print("PARAMETROS: ")
dir_realimentacion = sys.argv[1]  # /home/carloslinux/Desktop/GIT_BOLSA/BolsaJava/realimentacion/
##################################################################################################

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
    if "SG_0" in ficheroFP:
        print("El subgrupo SG_0 se excluye (Fichero " + ficheroFP + " )")
    if "SG_0" not in ficheroFP and os.path.exists(ficheroFP):
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
velasFP['fecha'] = pd.to_datetime(
    velasFP.rename(columns={'anio': 'year', 'mes': 'month', 'dia': 'day'}, inplace=False)[['year', 'month', 'day']])
velasFP['diaensemana'] = pd.DatetimeIndex(velasFP['fecha']).day_name()
# print("Ejemplos:"); print(velasFP.head())


##################################################################################################
# NUMERO DE PREDICCIONES CALCULADAS (TEST + VALIDACION) EN CADA SUBGRUPO
##################################################################################################
pathFilesPrediccionesTestYValid = []
for filename in glob.iglob(dirPasadoSubgrupos + '**/intermedio.csv.test_t_predicho.csv', recursive=True):
    pathFilesPrediccionesTestYValid.append(filename)
for filename in glob.iglob(dirPasadoSubgrupos + '**/intermedio.csv.validac_t_predicho.csv', recursive=True):
    pathFilesPrediccionesTestYValid.append(filename)

prediccionesTV = pd.DataFrame([], columns=["subgrupo", "numeroPredicciones"])
prediccionesTV_temp = pd.DataFrame([], columns=["indice", "targetpredicho"])
for ficheroPredTV in pathFilesPrediccionesTestYValid:
    sgExtraido = ficheroPredTV.split("/")[4]  # subgrupo
    if sgExtraido == "SG_0":
        print("El subgrupo " + sgExtraido + " se excluye (Fichero " + ficheroPredTV + " )")
    if sgExtraido != "SG_0" and os.path.exists(ficheroPredTV):
        prediccionesTV_temp = pd.read_csv(ficheroPredTV, sep="|")
        if prediccionesTV_temp.size > 0:
            prediccionesTV_temp.columns = ["indice", "targetpredicho"]  # porque viene sin cabecera
            numFilasTmp = prediccionesTV_temp.shape[0]
            nuevafila = {'subgrupo': sgExtraido, 'numeroPredicciones': numFilasTmp}
            prediccionesTV = prediccionesTV.append(nuevafila, ignore_index=True)

# Sumando los subtotales de Test y validacion por SUBGRUPO
prediccionesTV = prediccionesTV.groupby('subgrupo').sum()

##########################################################################################################
# NUMERO DE PREDICCIONES CALCULADAS (TEST + VALIDACION) EN CADA EMPRESA (de todos los subgrupos
# considerados como un conjunto unico)
##########################################################################################################
pathFilesPrediccionesTestYValidPorEmpresa = []
for filename in glob.iglob(dirPasadoSubgrupos + '**/todaslaspredicciones_TEST.csv', recursive=True):
    pathFilesPrediccionesTestYValidPorEmpresa.append(filename)
for filename in glob.iglob(dirPasadoSubgrupos + '**/todaslaspredicciones_VALIDACION.csv', recursive=True):
    pathFilesPrediccionesTestYValidPorEmpresa.append(filename)

prediccionesTodasTV = pd.DataFrame([], columns=["numeroPredicciones", "empresa", "subgrupo"])
prediccionesTodasTV_temp = pd.DataFrame([], columns=["subgrupo", "modo", "indice"])
for ficheroPredTV in pathFilesPrediccionesTestYValidPorEmpresa:
    sgExtraido = ficheroPredTV.split("/")[4]  # subgrupo
    if sgExtraido == "SG_0":
        print("El subgrupo " + sgExtraido + " se excluye (Fichero " + ficheroPredTV + " )")
    if sgExtraido != "SG_0" and os.path.exists(ficheroPredTV):
        prediccionesTodasTV_temp = pd.read_csv(ficheroPredTV, sep="|", header=None)
        if prediccionesTodasTV_temp.size > 0:
            prediccionesTodasTV_temp.columns = ["subgrupo", "modo", "indice"]  # porque viene sin cabecera
            prediccionesTodasTV_temp[['empresa', 'anio', 'mes', 'dia']] = prediccionesTodasTV_temp['indice'].str.split(
                '_', 4, expand=True)

            # Numero de predicciones total por cada empresa
            nuevasFilas = prediccionesTodasTV_temp.groupby(['empresa']).size()
            nuevasFilasDF = nuevasFilas.to_frame()
            nuevasFilasDF['empresa'] = nuevasFilasDF.index

            # Todas las empresas en esta iteracion, son de un mismo subgrupo
            nuevasFilasDF['subgrupo'] = sgExtraido

            nuevasFilasDF.columns = ["numeroPredicciones", "empresa", "subgrupo"]
            prediccionesTodasTV = prediccionesTodasTV.append(nuevasFilasDF, ignore_index=True)

empresasYsusSubgrupos_temp = prediccionesTodasTV  # para usarlo mas adelante: empresas y sus subgrupos

# Sumando los subtotales de Test y validacion por SUBGRUPO
prediccionesTodasTV = prediccionesTodasTV.groupby('empresa').sum()
prediccionesTodasTV = prediccionesTodasTV.drop('subgrupo', axis=1)
prediccionesTodasTV['empresa'] = prediccionesTodasTV.index
prediccionesTodasTV.reset_index(drop=True, inplace=True)

# Subgrupos en los que aparece cada empresa
empresasYsusSubgrupos = pd.DataFrame([], columns=["empresa", "subgrupo"])
empresasYsusSubgrupos_temp['subgrupos'] = empresasYsusSubgrupos_temp[['empresa', 'subgrupo']].groupby(['empresa'])[
    'subgrupo'].transform(lambda x: ';'.join(x))
empresasYsusSubgrupos = empresasYsusSubgrupos_temp[['empresa', 'subgrupos']].drop_duplicates()

# Empresa + numero total de predicciones + subgrupos en los que aparece
prediccionesTodasTV = prediccionesTodasTV.merge(empresasYsusSubgrupos, how='inner', on='empresa')

##########################################################################################################
##### ANALISIS ########
print("Ficheros CSV leidos (test y validation): " + str(ficherosFPContador))
print("Velas leidas (falsos positivos): " + str(velasFP.shape[0]))
print("Numero de empresas, anios, meses, etc distintos analizados: ")
velasFPunicos = velasFP.nunique().to_frame()
print(tabulate(velasFPunicos.transpose(), headers='keys', tablefmt='psql'))

numEmpresasAnalizadas = velasFPunicos.filter(items=['empresa'], axis=0)[0][0]

print("Top RATIO-EMPRESAS con MAS falsos positivos (para no invertir en ellas):")
data1 = velasFP.groupby('empresa')['dia'].count().to_frame().sort_values(by=['dia'], ascending=False)
data1.rename(columns={'dia': 'numvelasfp'}, inplace=True)
data1['empresa'] = data1.index
data1.reset_index(drop=True, inplace=True)
prediccionesTodasTV.reset_index(drop=True, inplace=True)
data2 = data1.merge(prediccionesTodasTV, how='inner', on='empresa')
data2['ratioFalsosPositivos'] = 100 * data2['numvelasfp'] / data2['numeroPredicciones']
data2 = data2.sort_values(by=['ratioFalsosPositivos'], ascending=True).round(1)
data2.reset_index(drop=True, inplace=True)
data2 = data2[
    ["numvelasfp", "empresa", "numeroPredicciones", "ratioFalsosPositivos", "subgrupos"]]  # reordenar columnas
print("FALSOSPOSITIVOS - EMPRESAS - Path: " + dirLogs + "falsospositivos_empresas.csv")
data2.to_csv(dirLogs + "falsospositivos_empresas.csv", index=True, sep='|', float_format='%.4f')
print(tabulate(data2.head(20).transpose(), headers='keys', tablefmt='psql'))

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

print("Top RATIO-SUBGRUPOS con MENOS falsos positivos:")
data1 = velasFP.groupby('subgrupo')['dia'].count().to_frame().sort_values(by=['dia'], ascending=True)
data1 = data1.rename(columns={"dia": "numvelasfp"})
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
print("Acumulación de inteligencia en la carpeta de realimentacion")


def acumularInteligencia(dirLogs, dir_realimentacion, nombrefichero, clave):
    print("Acumulando inteligencia del fichero: " + nombrefichero)
    pathOrigen = dirLogs + nombrefichero
    pathDestino = dir_realimentacion + nombrefichero

    # PENDIENTE mejorar

    # if os.path.exists(pathDestino):  # juntar info previa y actual
    #   previaDF = pd.read_csv(pathOrigen, delimiter="|")
    #   actualDF = pd.read_csv(pathDestino, delimiter="|")
    #   juntos = pd.concat([previaDF, actualDF])
    # else:  # pone la info actual
    shutil.copyfile(pathOrigen, pathDestino)
    # print(tabulate(juntos.head().to_frame().transpose(), headers='keys', tablefmt='psql'))


if numEmpresasAnalizadas > 500:
    print("Se han procesado suficientes empresas: " + str(
        numEmpresasAnalizadas) + "  Por tanto, se puede acumular conocimiento.")
    acumularInteligencia(dirLogs, dir_realimentacion, "falsospositivos_subgrupos.csv", "subgrupo")
    acumularInteligencia(dirLogs, dir_realimentacion, "falsospositivos_meses.csv", "mes")
    acumularInteligencia(dirLogs, dir_realimentacion, "falsospositivos_empresas.csv", "empresa")
    acumularInteligencia(dirLogs, dir_realimentacion, "falsospositivos_diaensemana.csv", "diaensemana")
else:
    print("No se han procesado suficientes empresas: " + str(
        numEmpresasAnalizadas) + "  Por tanto, NO se puede acumular conocimiento.")

print("============================================")
