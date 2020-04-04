#!/bin/bash

#################### DESCRIPCIÓN Y PARÁMETROS FUNDAMENTALES ###############################
# Permite sacar las predicciones en un rango de tiempos del pasado. El modelo predictor NO SERÁ el usado en 
# antigüedad =0, sino que cogeremos la antigüedad máxima deseada, y lo entrenaremos con datos anteriores.
# POR TANTO, LOS RESULTADOS PUEDEN DIFERIR DE LOS QUE SE USEN EN EL FUTURO CON DINERO REAL.
# Por otro lado, SIEMPRE SE DEBEN ACTIVAR LAS DESCARGAS (en parametros.config) para que, en cada 
# antigüedad analizada, se obtenga la fila de futuro a predecir con la fecha deseada.
# ¡¡¡ATENCIÓN: AL ACTIVAR LAS DESCARGAS, SE BORRARÁN LOS MODELOS ACTUALES!!!! GUARDARLOS PREVIAMENTE
# Es OBLIGATORIO que en la carpeta usada haya un fichero GRANDE con antiguedad 0 en su nombre, y 
# al menos X días más reciente que el rango a analizar.

#Instantes de las descargas
#Se analizará el tramo de antiguedad desde máxima hasta minima
ANTIGUEDAD_MAXIMA="25"
ANTIGUEDAD_MINIMA="15"


echo -e "INVERSION - INICIO: "$( date "+%Y%m%d%H%M%S" )

#################### DIRECTORIOS ###############################################################
DIR_CODIGOS_CARLOS="/home/carloslinux/Desktop/GIT_BOLSA/"
DIR_CODIGOS_LUIS="/home/t151521/bolsa/"
PYTHON_MOTOR_CARLOS="/home/carloslinux/Desktop/PROGRAMAS/anaconda3/envs/BolsaPython/bin/python"
PYTHON_MOTOR_LUIS="/home/t151521/anaconda3/envs/BolsaPython/bin/python"
DIR_DROPBOX_CARLOS="/home/carloslinux/Dropbox/BOLSA_PREDICTOR/"
DIR_DROPBOX_LUIS="/home/t151521/Dropbox/BOLSA_PREDICTOR/"

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

DIR_BASE="/bolsa/"
DIR_LOGS="${DIR_BASE}logs/"
LOG_INVERSION="${DIR_LOGS}inversion.log"
DIR_INVERSION="${DIR_BASE}inversion/"
DIR_FUT_SUBGRUPOS="${DIR_BASE}futuro/subgrupos/"
DIR_JAVA="${DIR_CODIGOS}BolsaJava/"
PATH_JAR="${DIR_JAVA}target/bolsajava-1.0-jar-with-dependencies.jar"
DIR_GITHUB_INVERSION="${DIR_CODIGOS}inversion/"

rm -Rf ${DIR_INVERSION}
crearCarpetaSiNoExiste "${DIR_INVERSION}"

############### LOGS ########################################################
rm -f "${DIR_LOGS}log4j.log"
rm -f "${LOG_INVERSION}"

#################################### CÓDIGO ###########################################################

 # Se obtiene el modelo de predicción para la antigüedad máxima. Luego se irá hacia adelante en el tiempo, prediciendo tiempos futuros para el modelo entrenado

${PATH_SCRIPTS}master.sh "pasado" "${ANTIGUEDAD_MAXIMA}" "0" "${ACTIVAR_DESCARGAS}" "S" "${S}" "${X}" "${R}" "${M}" "${F}" "${B}" "${NUM_EMPRESAS}" "${UMBRAL_SUBIDA_POR_VELA}" "${MIN_COBERTURA_CLUSTER}" "${MIN_EMPRESAS_POR_CLUSTER}" "${P_INICIO}" "${P_FIN}" "${MAX_NUM_FEAT_REDUCIDAS}" 2>>${LOG_INVERSION} 1>>${LOG_INVERSION}

# Análisis de inversión histórica. Se recorre cada antigüedad, y se predice su futuro
for (( ANTIGUEDAD=${ANTIGUEDAD_MINIMA}; ANTIGUEDAD<=${ANTIGUEDAD_MAXIMA}; ANTIGUEDAD++ ))
do  

rm -Rf /bolsa/futuro/ >>${LOG_INVERSION}
crearCarpetaSiNoExiste "${DIR_FUT_SUBGRUPOS}"


echo -e $( date '+%Y%m%d_%H%M%S' )" Ejecución del futuro (para velas de antiguedad=ANTIGUEDAD) con TODAS LAS EMPRESAS (lista DIRECTA ó INVERSA, ya da igual, no estamos mirando overfitting)..." >>${LOG_INVERSION}
MIN_COBERTURA_CLUSTER=0    # Para predecir, cojo lo que haya, sin minimos. EL modelo ya lo hemos entrenado
MIN_EMPRESAS_POR_CLUSTER=1   # Para predecir, cojo lo que haya, sin minimos. EL modelo ya lo hemos entrenado
${PATH_SCRIPTS}master.sh "futuro" "$ANTIGUEDAD" "0" "${ACTIVAR_DESCARGAS}" "S" "${S}" "${X}" "${R}" "${M}" "${F}" "${B}" "${NUM_EMPRESAS_INVERSION}" "${UMBRAL_SUBIDA_POR_VELA}" "${MIN_COBERTURA_CLUSTER}" "${MIN_EMPRESAS_POR_CLUSTER}" "20001111" "20991111" "${MAX_NUM_FEAT_REDUCIDAS}" 2>>${LOG_INVERSION} 1>>${LOG_INVERSION}

echo -e "Guardamos la prediccion del FUTURO de todos los SUBGRUPOS en la carpeta de INVERSION, para apostar dinero REAL..." >>${LOG_INVERSION}

while IFS= read -r -d '' -u 9
do
	if [[ $REPLY == *"COMPLETO_PREDICCION"* ]]; then
		echo "Procesamos  ${REPLY}  y lo copiamos en ${DIR_INVERSION} ..."  >>${LOG_INVERSION}
		ficheronombre=$(basename $REPLY)
		directorio=$(dirname $REPLY)
		$PYTHON_MOTOR "${PYTHON_SCRIPTS}bolsa/InversionUtils.py" "${directorio}/${ficheronombre}"  "0" "${DIR_DROPBOX}" "${ficheronombre}" >> ${LOG_INVERSION}
	fi
done 9< <( find ${DIR_FUT_SUBGRUPOS} -type f -exec printf '%s\0' {} + )

done


echo -e "INVERSION - FIN: "$( date "+%Y%m%d%H%M%S" )

