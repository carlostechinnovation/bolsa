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





