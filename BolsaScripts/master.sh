#!/bin/bash

################## PARAMETROS DE ENTRADA ############################################
DIR_TIEMPO="${1}" #pasado o futuro
DESPLAZAMIENTO_ANTIGUEDAD="${2}"  #0, -50 ....

################## FUNCIONES #############################################################
crearCarpetaSiNoExisteYVaciar() {
	param1=${1} 			#directorio
	echo "Creando carpeta: $param1"
	mkdir -p "${param1}"
	rm -f "${param1}*"
}

crearCarpetaSiNoExisteYVaciarRecursivo() {
	param1=${1} 			#directorio
	echo "Creando carpeta: $param1"
	mkdir -p "${param1}"
	rm -Rf "${param1}*"
}

################ VARIABLES DE EJECUCION #########################################################
ID_EJECUCION=$( date "+%Y%m%d%H%M%S" )
echo -e "ID_EJECUCION = "${ID_EJECUCION}

NUM_MAX_EMPRESAS_DESCARGADAS=1
MIN_COBERTURA_CLUSTER=60
MIN_EMPRESAS_POR_CLUSTER=10

# PARAMETROS DE TARGET MEDIDOS EN VELAS
S = 10
X = 28
R = 5
M = 7

#################### DIRECTORIOS ###############################################################
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
DIR_SUBGRUPOS="${DIR_BASE}${DIR_TIEMPO}/subgrupos/"
DIR_IMG="img/"

crearCarpetaSiNoExisteYVaciar "${DIR_BASE}"
crearCarpetaSiNoExisteYVaciar "${DIR_LOGS}"
crearCarpetaSiNoExisteYVaciar "${DIR_BASE}${DIR_TIEMPO}"
crearCarpetaSiNoExisteYVaciar "${DIR_BRUTOS}"
crearCarpetaSiNoExisteYVaciar "${DIR_BRUTOS_CSV}"
crearCarpetaSiNoExisteYVaciar "${DIR_LIMPIOS}"
crearCarpetaSiNoExisteYVaciar "${DIR_ELABORADOS}"
crearCarpetaSiNoExisteYVaciarRecursivo "${DIR_SUBGRUPOS}"


############### LOGS ########################################################
rm -f "${DIR_LOGS}log4j.log"
LOG_MASTER="${DIR_LOGS}${ID_EJECUCION}_bolsa_coordinador_${DIR_TIEMPO}.log"
rm -f "${LOG_MASTER}"

############### COMPILAR JAR ########################################################
echo -e "Compilando JAVA en un JAR..." >> ${LOG_MASTER}
cd "${DIR_JAVA}"
mvn clean compile assembly:single

################################################################################################
echo -e "-------- DATOS BRUTOS -------------" >> ${LOG_MASTER}

echo -e "DINAMICOS - Descargando de YAHOO FINANCE..." >> ${LOG_MASTER}
java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c10X.brutos.YahooFinance01Descargar" "${NUM_MAX_EMPRESAS_DESCARGADAS}" "${DIR_BRUTOS}" "${DIR_TIEMPO}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}

echo -e "DINAMICOS - Limpieza de YAHOO FINANCE..." >> ${LOG_MASTER}
java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c10X.brutos.YahooFinance02Parsear" "${DIR_BRUTOS}" "${DIR_BRUTOS_CSV}" "${DIR_TIEMPO}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}

echo -e "ESTATICOS - Descargando de FINVIZ (igual para Pasado o Futuro, salvo el directorio)..." >> ${LOG_MASTER}
java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c10X.brutos.EstaticosFinvizDescargarYParsear" "${NUM_MAX_EMPRESAS_DESCARGADAS}" "${DIR_BRUTOS}" "${DIR_BRUTOS_CSV}" "${ES_ENTORNO_VALIDACION}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}

echo -e "ESTATICOS + DINAMICOS: juntando en un CSV único..." >> ${LOG_MASTER}
java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c10X.brutos.JuntarEstaticosYDinamicosCSVunico" "${DIR_BRUTOS_CSV}" "${DESPLAZAMIENTO_ANTIGUEDAD}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}

echo -e "ESTATICOS + DINAMICOS: limpiando CSVs intermedios brutos..." >> ${LOG_MASTER}
java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c10X.brutos.LimpiarCSVBrutosTemporales" "${DIR_BRUTOS}" "${DIR_BRUTOS_CSV}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}


################################################################################################
echo -e "-------- DATOS LIMPIOS -------------" >> ${LOG_MASTER}

#######echo -e "Operaciones de limpieza: quitar outliers, rellenar missing values..." >> ${LOG_MASTER}
#######java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c30X.elaborados.LimpiarOperaciones" "${DIR_BRUTOS_CSV}" "${DIR_LIMPIOS}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}

# PENDIENTE: de momento, no limpiamos, pero habrá que hacerlo
cp ${DIR_BRUTOS_CSV}*.csv ${DIR_LIMPIOS} 2>>${LOG_MASTER} 1>>${LOG_MASTER}


################################################################################################
echo -e "-------- VARIABLES ELABORADAS -------------" >> ${LOG_MASTER}

echo -e "Calculando elaborados y target..." >> ${LOG_MASTER}
java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c30X.elaborados.ConstructorElaborados" "${DIR_LIMPIOS}" "${DIR_ELABORADOS}"  "${S}" "${X}" "${R}" "${M}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}

echo -e "Elaborados (incluye la variable elaborada TARGET) ya calculados" >> ${LOG_MASTER}


################################################################################################
echo -e "-------- SUBGRUPOS -------------" >> ${LOG_MASTER}

echo -e "Calculando subgrupos..." >> ${LOG_MASTER}
java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c40X.subgrupos.CrearDatasetsSubgrupos" "${DIR_ELABORADOS}" "${DIR_SUBGRUPOS}" "${MIN_COBERTURA_CLUSTER}" "${MIN_EMPRESAS_POR_CLUSTER}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}
#java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c40X.subgrupos.CrearDatasetsSubgruposKMeans" "${DIR_ELABORADOS}" "${DIR_SUBGRUPOS}" "${MIN_COBERTURA_CLUSTER}" "${MIN_EMPRESAS_POR_CLUSTER}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}


############  PARA CADA SUBGRUPO ###############################################################

for dir_subgrupo in ${DIR_SUBGRUPOS}*
do
	echo "-------- Subgrupo cuya carpeta es: ${dir_subgrupo} --------" >> ${LOG_MASTER}
	crearCarpetaSiNoExisteYVaciar  "${dir_subgrupo}/${DIR_IMG}"
	
	echo -e "Capa 5: elimina MISSING VALUES (NA en columnas y filas), elimina OUTLIERS, balancea clases (undersampling de mayoritaria), calcula IMG funciones de densidad, NORMALIZA las features, comprueba suficientes casos en clase minoritaria, REDUCCION de FEATURES y guarda el CSV REDUCIDO..." >> ${LOG_MASTER}
	python3 "${PYTHON_SCRIPTS}bolsa/C5NormalizarYReducirDatasetSubgrupo.py" "${dir_subgrupo}/" "${DIR_TIEMPO}" >> ${LOG_MASTER}
	
	echo -e "Capa 6 - PASADO ó FUTURO: balancea las clases (aunque ya se hizo en capa 5), divide dataset de entrada (entrenamiento, test, validación), CREA MODELOS (con hyperparámetros)  los evalúa. Guarda el modelo GANADOR de cada subgrupo..." >> ${LOG_MASTER}
	python3 "${PYTHON_SCRIPTS}bolsa/C6CreadorModelosDeSubgrupo.py" "${dir_subgrupo}/" "${DIR_TIEMPO}"  >> ${LOG_MASTER}
	
	echo -e "Capa 7: otras validaciones manuales (rentabilidad a posteriori, etc)..." >> ${LOG_MASTER}
	python3 "${PYTHON_SCRIPTS}bolsa/C7OtrasValidacionesManuales.py" >> ${LOG_MASTER}
	
	
done

############### BORRAR JAR ########################################################
rm -Rf "${DIR_JAVA}target/"

################################################################################################

