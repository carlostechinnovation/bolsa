#!/bin/bash

echo -e "INVERSION POSTERIORI - INICIO: "$( date "+%Y%m%d%H%M%S" )

#################### DIRECTORIOS ###############################################################
DIR_CODIGOS_CARLOS="/home/carloslinux/Desktop/GIT_BOLSA/"
DIR_CODIGOS_LUIS="/home/t151521/bolsa/"
PYTHON_MOTOR_CARLOS="/home/carloslinux/Desktop/PROGRAMAS/anaconda3/envs/BolsaPython/bin/python"
PYTHON_MOTOR_LUIS="/home/t151521/anaconda3/envs/BolsaPython/bin/python"

usuario=$(whoami)
if [ $usuario == "carloslinux" ]
then
  DIR_CODIGOS="${DIR_CODIGOS_CARLOS}"
  PYTHON_MOTOR="${PYTHON_MOTOR_CARLOS}"
elif [ $usuario == "t151521" ]
then
  DIR_CODIGOS="${DIR_CODIGOS_LUIS}"
  PYTHON_MOTOR="${PYTHON_MOTOR_LUIS}"
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

crearCarpetaSiNoExisteYVaciarRecursivo() {
	param1=${1} 			#directorio
	echo "Creando carpeta: $param1"
	mkdir -p ${param1}
	rm -Rf ${param1}*
}

################################################################################################
PATH_SCRIPTS="${DIR_CODIGOS}BolsaScripts/"
PYTHON_SCRIPTS="${DIR_CODIGOS}BolsaPython/"
PARAMS_CONFIG="${PATH_SCRIPTS}parametros.config"
echo -e "Importando parametros generales..."
source ${PARAMS_CONFIG}

#Instantes de las descargas
FUTURO_INVERSION="$((${X}+${M}))" #Instante del pasado en el que ejecuté inversion.sh

DIR_BASE="/bolsa/"
DIR_LOGS="${DIR_BASE}logs/"
LOG_INVERSIONPOST="${DIR_LOGS}inversion_posteriori.log"
DIR_INVERSION="${DIR_BASE}inversion/"
DIR_INVERSIONPOST="${DIR_BASE}inversion_posteriori/"
DIR_FUT_SUBGRUPOS="${DIR_BASE}futuro/subgrupos/"
DIR_JAVA="${DIR_CODIGOS}BolsaJava/"
PATH_JAR="${DIR_JAVA}target/bolsajava-1.0-jar-with-dependencies.jar"
DIR_GITHUB_INVERSIONPOST="${DIR_CODIGOS}inversion_posteriori/"

rm -Rf ${DIR_INVERSION}
crearCarpetaSiNoExiste "${DIR_INVERSION}"

############### LOGS ########################################################
rm -f "${DIR_LOGS}log4j.log"
rm -f "${LOG_INVERSION}"

################################################################################################

#En esta ejecucion SOLO NOS INTERESA COGER el tablon analítico de entrada, para extraer el resultado REAL y compararlo con las predicciones que hicimos unas velas atrás.
#En esta ejecucion, no miramos la predicción. --> Esta NO es la predicción del futuro en el instante de ahora mismo (t1=0) para poner dinero real, sino de del futuro de que hemos predicho en INVERSION.SH, es decir, en ANTIGUEDAD=X+M 

rm -Rf /bolsa/futuro/ >>${LOG_INVERSIONPOST}
crearCarpetaSiNoExiste "${DIR_FUT_SUBGRUPOS}"

echo -e $( date '+%Y%m%d_%H%M%S' )" Ejecución del futuro (para velas de antiguedad=${FUTURO2_t1}) con OTRAS empresas (lista REVERTIDA)..." >>${LOG_INVERSIONPOST}
${PATH_SCRIPTS}master.sh "futuro" "${FUTURO_INVERSION}" "0" "S" "S" "${S}" "${X}" "${R}" "${M}" "${F}" "${B}" "${NUM_EMPRESAS_INVERSION}" "${UMBRAL_SUBIDA_POR_VELA}" "${MIN_COBERTURA_CLUSTER}" "${MIN_EMPRESAS_POR_CLUSTER}" "20001111" "20991111" "${MAX_NUM_FEAT_REDUCIDAS}" 2>>${LOG_INVERSIONPOST} 1>>${LOG_INVERSIONPOST}

echo -e "Guardamos el resultado REAL que ha sucedido..." >>${LOG_INVERSIONPOST}

HOY_YYYYMMDD=$( date "+%Y%m%d" )

while IFS= read -r -d '' -u 9
do
	if [[ $REPLY == *"COMPLETO_PREDICCION"* ]]; then
		echo "Copiando este fichero   $REPLY  en  ${DIR_INVERSION} ..."  >>${LOG_INVERSION}
		ficheronombre=$(basename $REPLY)
		directorio=$(dirname $REPLY)
		mv "${directorio}/${ficheronombre}" "${directorio}/${HOY_YYYYMMDD}${ficheronombre}"
		cp "${directorio}/${HOY_YYYYMMDD}${ficheronombre}" "${DIR_INVERSION}"
	fi
done 9< <( find ${DIR_FUT_SUBGRUPOS} -type f -exec printf '%s\0' {} + )


################# PENDIENTE #####################

# SOLO queremos extraer lo que ha pasado . Es decir, irse a la vela con antiguedad X+M y coger la columna "TARGET REAL"
#
# Mapear lo predicho (inversion.sh) con lo real (este script), GENERANDO un CSV con todo y poniendolo aqui: ${DIR_INVERSIONPOST}
#
#
#

################# VOLCADO EN EL HISTORICO GITHUB (que no borraremos nunca) #########################################################
cp -Rf "${DIR_INVERSIONPOST}" $DIR_GITHUB_INVERSIONPOST




echo -e "INVERSION POSTERIORI - FIN: "$( date "+%Y%m%d%H%M%S" )