#!/bin/bash

echo -e "INVERSION POSTERIORI - INICIO: "$( date "+%Y%m%d%H%M%S" )

#################### DIRECTORIOS ###############################################################
DIR_DROPBOX_CARLOS="/home/carloslinux/Dropbox/BOLSA_PREDICTOR/"
DIR_DROPBOX_LUIS="/home/t151521/Dropbox/BOLSA_PREDICTOR/"
PYTHON_MOTOR_CARLOS="/home/carloslinux/Desktop/PROGRAMAS/anaconda3/envs/BolsaPython/bin/python"
PYTHON_MOTOR_LUIS="/home/t151521/anaconda3/envs/BolsaPython/bin/python"

usuario=$(whoami)
if [ $usuario == "carloslinux" ]
then
  DIR_DROPBOX="${DIR_DROPBOX_CARLOS}"
  PYTHON_MOTOR="${PYTHON_MOTOR_CARLOS}"
elif [ $usuario == "t151521" ]
then
  DIR_DROPBOX="${DIR_DROPBOX_LUIS}"
  PYTHON_MOTOR="${PYTHON_MOTOR_LUIS}"
else
  echo "ERROR: USUARIO NO CONTROLADO"
  s1 -e
fi

################## PARAMETROS DE ENTRADA ############################################
# PARAMETROS DE TARGET MEDIDOS EN VELAS
S="${1}" #default=10
X="${2}" #default=56
R="${3}" #default=10
M="${4}" #default=7
F="${5}" #default=5
B="${6}" #default=5

################## FUNCIONES #############################################################
crearCarpetaSiNoExiste() {
	param1=${1} 			#directorio
	echo "Creando carpeta: $param1"
	mkdir -p ${param1}
}

crearCarpetaSiNoExisteYVaciar() {
	param1=${1} 			#directorio
	echo "Creando carpeta: $param1"
	mkdir -p ${param1}
	rm -f ${param1}*
}

crearCarpetaSiNoExisteYVaciarRecursivo() {
	param1=${1} 			#directorio
	echo "Creando carpeta: $param1"
	mkdir -p ${param1}
	rm -Rf ${param1}*
}

################ VARIABLES DE EJECUCION #########################################################
ID_EJECUCION=$( date "+%Y%m%d%H%M%S" )
echo -e "ID_EJECUCION = "${ID_EJECUCION}

PATH_SCRIPTS="${DIR_CODIGOS}BolsaScripts/"
PYTHON_SCRIPTS="${DIR_CODIGOS}BolsaPython/"
DIR_JAVA="${DIR_CODIGOS}BolsaJava/"
PATH_JAR="${DIR_JAVA}target/bolsajava-1.0-jar-with-dependencies.jar"

DIR_BASE="/bolsa/"
DIR_ANALISIS="${DIR_DROPBOX}ANALISIS/"
DIR_BRUTOS_CSV="${DIR_BASE}futuro/brutos_csv/"

crearCarpetaSiNoExiste "${DIR_ANALISIS}"

############### LOGS ########################################################
rm -f "${DIR_LOGS}log4j.log"
LOG_MASTER="${DIR_LOGS}${ID_EJECUCION}_bolsa_coordinador_analisis.log"
rm -f "${LOG_MASTER}"

############### COMPILAR JAR ########################################################
echo -e "Compilando JAVA en un JAR..." >> ${LOG_MASTER}
cd "${DIR_JAVA}" >> ${LOG_MASTER}
rm -Rf "${DIR_JAVA}target/" >> ${LOG_MASTER}
mvn clean compile assembly:single >> ${LOG_MASTER}

################################################################################################
$PYTHON_MOTOR "${PYTHON_SCRIPTS}bolsa/InversionUtilsPosteriori.py" "${DIR_DROPBOX}" "${DIR_ANALISIS}" "${S}" "${X}" "${R}" "${M}" "${F}" "${B}" >> ${LOG_MASTER}

echo -e "INVERSION POSTERIORI - FIN: "$( date "+%Y%m%d%H%M%S" )








