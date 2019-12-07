#!/bin/bash

ID_EJECUCION=$( date '+%Y%m%d%H%M%S' )
echo -e "ID_EJECUCION = "${ID_EJECUCION}


DIR_BASE="C:\DATOS\GITHUB_REPOS\bolsa\"
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


################################################################################################
echo -e "-------- DATOS LIMPIOS -------------" >> ${LOG_MASTER}




################################################################################################
echo -e "-------- VARIABLES ELABORADAS -------------" >> ${LOG_MASTER}
DIR_ELABORADOS="/bolsa/pasado/elaborados/"
DIR_ELABORADOS_CSV="/bolsa/pasado/elaborados_csv/"

echo -e "Calculando elaborados y target..." >> ${LOG_MASTER}
java -Djava.util.logging.SimpleFormatter.format='%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n' -jar ${PATH_JAR} --class 'c30X.elaborados.ConstructorElaborados' '${DIR_BRUTOS}' '${DIR_BRUTOS_CSV}' 2>>${PATH_LOG} 1>>${PATH_LOG}

echo -e "Elaborados y target ya calculados" >> ${LOG_MASTER}


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





