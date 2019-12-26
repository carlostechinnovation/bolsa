#!/bin/bash

################ VARIABLES DE EJECUCION #########################################################
ID_EJECUCION=$( date "+%Y%m%d%H%M%S" )
echo -e "ID_EJECUCION = "${ID_EJECUCION}

MODO="P" #Pasado (P) o Futuro (F)
DIR_TIEMPO=""
if [ "${MODO}" = "P" ]; then 
    DIR_TIEMPO="pasado"
elif [ "${MODO}" = "F" ]; then
    DIR_TIEMPO="futuro"
else
    echo "Debes elegir el modo: Pasado o Futuro"
	exit -1
fi;

MIN_COBERTURA_CLUSTER=60
MIN_EMPRESAS_POR_CLUSTER=10

#################### DIRECTORIOS ###############################################################
#DIR_CODIGOS="C:\DATOS\GITHUB_REPOS\bolsa\"
DIR_CODIGOS="/home/carloslinux/Desktop/GIT_BOLSA/"
PATH_SCRIPTS="${DIR_CODIGOS}BolsaScripts/"
PYTHON_SCRIPTS="${DIR_CODIGOS}BolsaPython/"
DIR_JAVA="${DIR_CODIGOS}BolsaJava/"
PATH_JAR="${DIR_JAVA}target/bolsajava-1.0-jar-with-dependencies.jar"

DIR_BASE="/bolsa/"
DIR_LOGS="${DIR_BASE}logs/"
DIR_BRUTOS="${DIR_BASE}${DIR_TIEMPO}/brutos/"
DIR_BRUTOS_CSV="${DIR_BASE}${DIR_TIEMPO}/brutos_csv/"
DIR_LIMPIOS="${DIR_BASE}${DIR_TIEMPO}/limpios/"
DIR_ELABORADOS="${DIR_BASE}${DIR_TIEMPO}/elaborados/"
DIR_SUBGRUPOS="${DIR_BASE}${DIR_TIEMPO}/datasets/"
DIR_MODELOS="${DIR_BASE}modelos/"
DIR_SUBGRUPOS_REDUCIDOS="${DIR_SUBGRUPOS}reducidos/"
DIR_SUBGRUPOS_IMG="${DIR_SUBGRUPOS}img/"

mkdir -p "${DIR_BASE}"
mkdir -p "${DIR_LOGS}"
mkdir -p "${DIR_BRUTOS}"
mkdir -p "${DIR_BRUTOS_CSV}"
rm -f "${DIR_BRUTOS}YF*.txt"
rm -f "${DIR_BRUTOS}YF*.csv"
mkdir -p "${DIR_LIMPIOS}"
mkdir -p "${DIR_ELABORADOS}"
mkdir -p "${DIR_SUBGRUPOS}"
mkdir -p "${DIR_MODELOS}"


############### LOGS ########################################################
rm -f "${DIR_LOGS}log4j.log"
LOG_MASTER="${DIR_LOGS}${ID_EJECUCION}_bolsa_coordinador_${MODO}.log"
rm -f "${LOG_MASTER}"

############### COMPILAR JAR ########################################################
echo -e "Compilando JAVA en un JAR..." >> ${LOG_MASTER}
cd "${DIR_JAVA}"
mvn clean compile assembly:single

################################################################################################
echo -e "-------- DATOS BRUTOS -------------" >> ${LOG_MASTER}

############## echo -e "Descargando de NASDAQ-OLD..." >> ${LOG_MASTER}
############## java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c10X.brutos.EstaticosNasdaqDescargarYParsear" "${DIR_BRUTOS}" "${DIR_BRUTOS_CSV}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}

echo -e "DINAMICOS - Descargando de YAHOO FINANCE..." >> ${LOG_MASTER}
java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c10X.brutos.YahooFinance01Descargar" "2" "${DIR_BRUTOS}" "P" 2>>${LOG_MASTER} 1>>${LOG_MASTER}

echo -e "DINAMICOS - Limpieza de YAHOO FINANCE..." >> ${LOG_MASTER}
java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c10X.brutos.YahooFinance02Parsear" "${DIR_BRUTOS}" "${DIR_BRUTOS_CSV}" "P" 2>>${LOG_MASTER} 1>>${LOG_MASTER}

echo -e "ESTATICOS - Descargando de FINVIZ (igual para Pasado o Futuro, salvo el directorio)..." >> ${LOG_MASTER}
java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c10X.brutos.EstaticosFinvizDescargarYParsear" "2" "${DIR_BRUTOS}" "${DIR_BRUTOS_CSV}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}

echo -e "ESTATICOS + DINAMICOS: juntando en un CSV único..." >> ${LOG_MASTER}
java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c10X.brutos.JuntarEstaticosYDinamicosCSVunico" "${DIR_BRUTOS}" "${DIR_BRUTOS_CSV}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}

echo -e "ESTATICOS + DINAMICOS: limpiando CSVs intermedios brutos..." >> ${LOG_MASTER}
java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c10X.brutos.LimpiarCSVBrutosTemporales" "${DIR_BRUTOS}" "${DIR_BRUTOS_CSV}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}


################################################################################################
echo -e "-------- DATOS LIMPIOS -------------" >> ${LOG_MASTER}

#######echo -e "Operaciones de limpieza: quitar outliers, rellenar missing values..." >> ${LOG_MASTER}
#######java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c30X.elaborados.LimpiarOperaciones" "${DIR_BRUTOS_CSV}" "${DIR_LIMPIOS}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}

# PENDIENTE: de momento, no limpiamos, pero habrá que hacerlo
cp "${DIR_BRUTOS_CSV}*.csv" "${DIR_LIMPIOS}"


################################################################################################
echo -e "-------- VARIABLES ELABORADAS -------------" >> ${LOG_MASTER}

echo -e "Calculando elaborados y target..." >> ${LOG_MASTER}
java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c30X.elaborados.ConstructorElaborados" "${DIR_LIMPIOS}" "${DIR_ELABORADOS}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}

echo -e "Elaborados (incluye la variable elaborada TARGET) ya calculados" >> ${LOG_MASTER}


################################################################################################
echo -e "-------- SUBGRUPOS -------------" >> ${LOG_MASTER}

echo -e "Calculando subgrupos..." >> ${LOG_MASTER}
java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c40X.subgrupos.CrearDatasetsSubgrupos" "${DIR_ELABORADOS}" "${DIR_SUBGRUPOS}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}
#java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c40X.subgrupos.CrearDatasetsSubgruposKMeans" "${DIR_ELABORADOS}" "${DIR_SUBGRUPOS}" "${MIN_COBERTURA_CLUSTER}" "${MIN_EMPRESAS_POR_CLUSTER}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}


############  PARA CADA SUBGRUPO ###############################################################

for path_csv_subgrupo in "${DIR_SUBGRUPOS}"/*
do
	echo "Analizando subgrupo cuyo dataset de entrada es: ${path_csv_subgrupo}" >> ${LOG_MASTER}
	mkdir -p "${DIR_SUBGRUPOS_REDUCIDOS}"
	mkdir -p "${DIR_SUBGRUPOS_IMG}"
	
	echo -e "-------- PARA CADA SUBGRUPO: SELECCIÓN DE VARIABLES -------------" >> ${LOG_MASTER}
	python3 "${PYTHON_SCRIPTS}bolsa/C5NormalizarYReducirDatasetSubgrupo.py" "${path_csv_subgrupo}" "${DIR_SUBGRUPOS_REDUCIDOS}"  "${DIR_SUBGRUPOS_IMG}"
	
	echo -e "-------- PARA CADA SUBGRUPO: CREACIÓN DE MODELOS (entrenamiento, test, validación) -------------" >> ${LOG_MASTER}
	python3 "${PYTHON_SCRIPTS}bolsa/C6CreadorModelosDeSubgrupo.py" "${DIR_SUBGRUPOS_REDUCIDOS}" "${DIR_MODELOS}"
	
	echo -e "-------- PARA CADA SUBGRUPO: EVALUACIÓN DE MODELOS (ROC, R2...); GUARDAR MODELO GANADOR -------------" >> ${LOG_MASTER}
	python3 "${PYTHON_SCRIPTS}bolsa/C7EvaluadorModelosDeSubgrupo.py"
	
	echo -e "-------- PARA CADA SUBGRUPO: VALIDACIÓN MANUAL DE MODELO GANADOR (rentabilidad, etc) -------------" >> ${LOG_MASTER}
	python3 "${PYTHON_SCRIPTS}bolsa/C8OtrasValidacionesManuales.py"
	
	
done

################################################################################################

############### BORRAR JAR ########################################################
#rm -Rf "${DIR_JAVA}target/"

################################################################################################

