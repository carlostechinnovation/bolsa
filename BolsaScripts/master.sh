#!/bin/bash

ID_EJECUCION=$( date '+%Y%m%d%H%M%S' )
echo -e "ID_EJECUCION = "${ID_EJECUCION}


DIR_BASE="/bolsa/"
LOG_MASTER="${DIR_BASE}bolsa_coordinador.log"
PATH_SCRIPTS="C:\DATOS\GITHUB_REPOS\bolsa\BolsaScripts/"
PYTHON_SCRIPTS="C:\DATOS\GITHUB_REPOS\bolsa\BolsaPython/"
PATH_JAR="C:\DATOS\GITHUB_REPOS\bolsa\BolsaJava/target/bolsajava-1.0.jar"

mkdir -p "${DIR_BASE}"

#Limpiar logs
rm -f "${DIR_BASE}../../bolsa_log4j.log"
rm -f "${LOG_MASTER}"


################################################################################################
echo -e "-------- DATOS BRUTOS -------------" >> ${LOG_MASTER}
DIR_BRUTOS="/bolsa/pasado/brutos/"
DIR_BRUTOS_CSV="/bolsa/pasado/brutos_csv/"

mkdir -p "${DIR_BRUTOS}"
mkdir -p "${DIR_BRUTOS_CSV}"

############## echo -e "Descargando de NASDAQ-OLD..." >> ${LOG_MASTER}
############## java -Djava.util.logging.SimpleFormatter.format='%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n' -jar ${PATH_JAR} --class 'c10X.brutos.EstaticosNasdaqDescargarYParsear' '${DIR_BRUTOS}' '${DIR_BRUTOS_CSV}' 2>>${PATH_LOG} 1>>${PATH_LOG}

echo -e "Descargando de YAHOO FINANCE..." >> ${LOG_MASTER}
java -Djava.util.logging.SimpleFormatter.format='%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n' -jar ${PATH_JAR} --class 'c10X.brutos.YahooFinance01Descargar' '2' '${DIR_BRUTOS}' '${DIR_BRUTOS_CSV}' 2>>${PATH_LOG} 1>>${PATH_LOG}

echo -e "Limpieza de YAHOO FINANCE..." >> ${LOG_MASTER}
java -Djava.util.logging.SimpleFormatter.format='%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n' -jar ${PATH_JAR} --class 'c10X.brutos.YahooFinance02Parsear' '${DIR_BRUTOS}' '${DIR_BRUTOS_CSV}' 2>>${PATH_LOG} 1>>${PATH_LOG}

echo -e "Descargando de FINVIZ..." >> ${LOG_MASTER}
java -Djava.util.logging.SimpleFormatter.format='%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n' -jar ${PATH_JAR} --class 'c10X.brutos.EstaticosFinvizDescargarYParsear' '2' '${DIR_BRUTOS}' '${DIR_BRUTOS_CSV}' 2>>${PATH_LOG} 1>>${PATH_LOG}

echo -e "Juntando en un CSV único: ESTATICOS y DINAMICOS..." >> ${LOG_MASTER}
java -Djava.util.logging.SimpleFormatter.format='%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n' -jar ${PATH_JAR} --class 'c10X.brutos.JuntarEstaticosYDinamicosCSVunico' '${DIR_BRUTOS}' '${DIR_BRUTOS_CSV}' 2>>${PATH_LOG} 1>>${PATH_LOG}

echo -e "Limpiando CSVs intermedios brutos..." >> ${LOG_MASTER}
java -Djava.util.logging.SimpleFormatter.format='%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n' -jar ${PATH_JAR} --class 'c10X.brutos.LimpiarCSVBrutosTemporales' '${DIR_BRUTOS}' '${DIR_BRUTOS_CSV}' 2>>${PATH_LOG} 1>>${PATH_LOG}


################################################################################################
echo -e "-------- DATOS LIMPIOS -------------" >> ${LOG_MASTER}
DIR_LIMPIOS="/bolsa/pasado/limpios/"
mkdir -p "${DIR_LIMPIOS}"

#######echo -e "Operaciones de limpieza: quitar outliers, rellenar missing values..." >> ${LOG_MASTER}
#######java -Djava.util.logging.SimpleFormatter.format='%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n' -jar ${PATH_JAR} --class 'c30X.elaborados.LimpiarOperaciones' '${DIR_BRUTOS_CSV}' '${DIR_LIMPIOS}' 2>>${PATH_LOG} 1>>${PATH_LOG}

# PENDIENTE: de momento, no limpiamos, pero habrá que hacerlo
cp '${DIR_BRUTOS_CSV}*' '${DIR_LIMPIOS}' 


################################################################################################
echo -e "-------- VARIABLES ELABORADAS -------------" >> ${LOG_MASTER}
DIR_ELABORADOS="/bolsa/pasado/elaborados/"
mkdir -p "${DIR_ELABORADOS}"

echo -e "Calculando elaborados y target..." >> ${LOG_MASTER}
java -Djava.util.logging.SimpleFormatter.format='%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n' -jar ${PATH_JAR} --class 'c30X.elaborados.ConstructorElaborados' '${DIR_LIMPIOS}' '${DIR_ELABORADOS}' 2>>${PATH_LOG} 1>>${PATH_LOG}

echo -e "Elaborados (incluye la variable elaborada TARGET) ya calculados" >> ${LOG_MASTER}


################################################################################################
echo -e "-------- SUBGRUPOS -------------" >> ${LOG_MASTER}
DIR_SUBGRUPOS="/bolsa/pasado/datasets/"
mkdir -p "${DIR_SUBGRUPOS}"

echo -e "Calculando subgrupos..." >> ${LOG_MASTER}
java -Djava.util.logging.SimpleFormatter.format='%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n' -jar ${PATH_JAR} --class 'c40X.subgrupos.CrearDatasetsSubgrupos' '${DIR_ELABORADOS}' '${DIR_SUBGRUPOS}' 2>>${PATH_LOG} 1>>${PATH_LOG}

echo -e "Subgrupos ya generados" >> ${LOG_MASTER}



############  PARA CADA SUBGRUPO ###############################################################


for path_csv_subgrupo in "${DIR_SUBGRUPOS}"/*
do
	echo "Analizando subgrupo cuyo dataset de entrada es: ${path_csv_subgrupo}"
	
	echo -e "-------- PARA CADA SUBGRUPO: SELECCIÓN DE VARIABLES -------------" >> ${LOG_MASTER}
	mkdir -p "${PYTHON_SCRIPTS}bolsa/C5SeleccionDeVariablesDeSubgrupo.py" "${path_csv_subgrupo}"
	
	echo -e "-------- PARA CADA SUBGRUPO: CREACIÓN DE MODELOS (entrenamiento, test, validación) -------------" >> ${LOG_MASTER}
	DIR_MODELOS="/bolsa/modelos/"
	mkdir -p "${PYTHON_SCRIPTS}bolsa/C6CreadorModelosDeSubgrupo.py" "${path_csv_subgrupo}" "${DIR_MODELOS}"
	
	echo -e "-------- PARA CADA SUBGRUPO: EVALUACIÓN DE MODELOS (ROC, R2...); GUARDAR MODELO GANADOR -------------" >> ${LOG_MASTER}
	python "${PYTHON_SCRIPTS}bolsa/C7EvaluadorModelosDeSubgrupo.py"
	
	echo -e "-------- PARA CADA SUBGRUPO: VALIDACIÓN MANUAL DE MODELO GANADOR (rentabilidad, etc) -------------" >> ${LOG_MASTER}
	python "${PYTHON_SCRIPTS}bolsa/C8OtrasValidacionesManuales.py"
	
	
	
	
done

################################################################################################
################################################################################################
echo -e "-------- CADENA FUTURA -------------" >> ${LOG_MASTER}





