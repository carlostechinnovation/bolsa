#!/bin/bash

ID_EJECUCION=$( date '+%Y%m%d%H%M%S' )
echo -e "ID_EJECUCION = "${ID_EJECUCION}


DIR_BASE="/home/carloslinux/git/bolsa/"
LOG_MASTER="${DIR_BASE}../../bolsa_coordinador.log"
PATH_SCRIPTS="${DIR_BASE}BolsaScripts/"
PATH_JAR="${DIR_BASE}BolsaJava/target/bolsajava-1.0.jar"


#Limpiar logs
rm -f "${DIR_BASE}../../bolsa_log4j.log"
rm -f "${LOG_MASTER}"



################################################################################################
echo -e "-------- DATOS BRUTOS -------------" >> ${LOG_MASTER}
DIR_BRUTOS="/bolsa/pasado/brutos/"
DIR_BRUTOS_CSV="/bolsa/pasado/brutos_csv/"


echo -e "Descargando..." >> ${LOG_MASTER}
java -Djava.util.logging.SimpleFormatter.format='%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n' -jar ${PATH_JAR} --class 'c10X.brutos.YahooFinance01Descargar' 'NASDAQ' 'AXAS' '${DIR_BRUTOS}' 2>>${PATH_LOG} 1>>${PATH_LOG}

echo -e "Estructurando (primera limpieza)..." >> ${LOG_MASTER}
java -Djava.util.logging.SimpleFormatter.format='%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n' -jar ${PATH_JAR} --class 'c10X.brutos.YahooFinance02Parsear' '${DIR_BRUTOS}\bruto_NASDAQ_AXAS.txt' '${DIR_BRUTOS_CSV}\bruto_NASDAQ_AXAS.csv' 2>>${PATH_LOG} 1>>${PATH_LOG}


echo -e "Juntando en un CSV único..." >> ${LOG_MASTER}

################################################################################################
echo -e "-------- DATOS LIMPIOS -------------" >> ${LOG_MASTER}




################################################################################################
echo -e "-------- VARIABLES ELABORADAS -------------" >> ${LOG_MASTER}



################################################################################################
echo -e "-------- SUBGRUPOS -------------" >> ${LOG_MASTER}



################################################################################################
echo -e "-------- PARA CADA SUBGRUPO: SELECCIÓN DE VARIABLES -------------" >> ${LOG_MASTER}



################################################################################################
echo -e "-------- PARA CADA SUBGRUPO: CREACIÓN DE MODELOS (entrenamiento, test, validación) -------------" >> ${LOG_MASTER}



################################################################################################
echo -e "-------- PARA CADA SUBGRUPO: EVALUACIÓN DE MODELOS (ROC, R2...); GUARDAR MODELO GANADOR -------------" >> ${LOG_MASTER}



################################################################################################
echo -e "-------- PARA CADA SUBGRUPO: VALIDACIÓN MANUAL DE MODELO GANADOR (rentabilidad, etc) -------------" >> ${LOG_MASTER}



################################################################################################
echo -e "-------- CADENA FUTURA -------------" >> ${LOG_MASTER}





