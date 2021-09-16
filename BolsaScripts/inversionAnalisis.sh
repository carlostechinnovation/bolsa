#!/bin/bash

#set -e

echo -e "INVERSION POSTERIORI - INICIO: "$( date "+%Y%m%d%H%M%S" )

#################### DIRECTORIOS ###############################################################
DIR_BASE="/bolsa/"
DIR_DROPBOX_REPO="/BOLSA_PREDICTOR/"
PYTHON_MOTOR_CARLOS="/home/carloslinux/Desktop/PROGRAMAS/anaconda3/envs/BolsaPython/bin/python"
PYTHON_MOTOR_LUIS="/home/t151521/anaconda3/envs/BolsaPython/bin/python"
DIR_CODIGOS_CARLOS="/home/carloslinux/Desktop/GIT_BOLSA/"
DIR_CODIGOS_LUIS="/home/t151521${DIR_BASE}"
DIR_DROPBOX_CARLOS="/home/carloslinux/Dropbox${DIR_DROPBOX_REPO}"
DIR_DROPBOX_LUIS="/home/t151521/Dropbox${DIR_DROPBOX_REPO}"

usuario=$(whoami)
if [ $usuario == "carloslinux" ]
then
  DIR_CODIGOS="${DIR_CODIGOS_CARLOS}"
  PYTHON_MOTOR="${PYTHON_MOTOR_CARLOS}"
  DIR_DROPBOX="${DIR_DROPBOX_CARLOS}"
elif [ $usuario == "t151521" ]
then
  DIR_CODIGOS="${DIR_CODIGOS_LUIS}"
  PYTHON_MOTOR="${PYTHON_MOTOR_LUIS}"
  DIR_DROPBOX="${DIR_DROPBOX_LUIS}"
else
  echo "ERROR: USUARIO NO CONTROLADO"
  s1 -e
fi

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
PATH_SCRIPTS="${DIR_CODIGOS}BolsaScripts/"
PYTHON_SCRIPTS="${DIR_CODIGOS}BolsaPython/"
PARAMS_CONFIG="${PATH_SCRIPTS}parametros.config"
echo -e "Importando parametros generales..."
source ${PARAMS_CONFIG}

DIR_JAVA="${DIR_CODIGOS}BolsaJava/"
PATH_JAR="${DIR_JAVA}target/bolsajava-1.0-jar-with-dependencies.jar"

DIR_LOGS="${DIR_BASE}logs/"
DIR_ANALISIS="${DIR_DROPBOX}ANALISIS/"
DIR_BRUTOS_CSV="${DIR_BASE}futuro/brutos_csv/"

crearCarpetaSiNoExiste "${DIR_ANALISIS}"

############### LOGS ########################################################
rm -f "${DIR_LOGS}log4j.log"
LOG_MASTER="${DIR_LOGS}${ID_EJECUCION}_bolsa_coordinador_analisis.log"
rm -f "${LOG_MASTER}"

################################################################################################
$PYTHON_MOTOR "${PYTHON_SCRIPTS}bolsa/InversionUtilsPosteriori.py" "${DIR_DROPBOX}" "${DIR_ANALISIS}" "${S}" "${X}" "${R}" "${M}" "${F}" "${B}" >> ${LOG_MASTER}

echo -e "INVERSION POSTERIORI - FIN: "$( date "+%Y%m%d%H%M%S" )








