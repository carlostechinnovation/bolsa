#!/bin/bash

ID_EJECUCION=$( date '+%Y%m%d%H%M%S' )
echo -e "ID_EJECUCION = "${ID_EJECUCION}


DIR_BASE="/home/carloslinux/Desktop/BOLSA/"
LOG_MASTER="${DIR_BASE}coordinador.log"
PATH_SCRIPTS="/home/carloslinux/git/bolsa/BolsaScripts/"

#Limpiar logs
rm -f "/home/carloslinux/Desktop/LOGS/log4j-application.log"
rm -f "${LOG_MASTER}"



################################################################################################
echo -e "-------- DATOS BRUTOS -------------" >> ${LOG_MASTER}

PATH_NASDAQ_TICKERS="C:\DATOS\GITHUB_REPOS\bolsa\knime_mockdata\nasdaq_tickers.csv"
PATH_JAR="/home/carloslinux/Desktop/GIT_REPO_BDML/bdml/mod002parser/target/mod002parser-jar-with-dependencies.jar"

FILE_BOE_OUT="/home/carloslinux/Desktop/DATOS_BRUTO/bolsa/BOE_out"
rm ${FILE_BOE_OUT}
java -jar ${PATH_JAR} "01" -Djava.util.logging.SimpleFormatter.format='%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n' 2>>${PATH_LOG} 1>>${PATH_LOG}


echo -e "Descargando..." >> ${LOG_MASTER}


echo -e "Limpiando..." >> ${LOG_MASTER}


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





