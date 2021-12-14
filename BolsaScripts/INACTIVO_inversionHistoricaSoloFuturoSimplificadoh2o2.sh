#!/bin/bash

#set -e

#################### DESCRIPCIÓN Y PARÁMETROS FUNDAMENTALES ###############################
# Permite sacar las predicciones en un rango de tiempos del pasado. 
# Se coge rel modelo de la matigüedad máxima (no lo genera este script). Luego se predice el futuro de la antigüedad mínima, y
# se calcula también el de las antigüedades intermedias en base a lo ya descargado para la antigüedad mínima (fichero COMPLETO.csv), 
# borrando previamente los días necesarios en le fichero COMPLETO y prediciendo el futuro sin redescargar.
# El modelo predictor NO SERÁ el usado en 
# antigüedad =0, sino que cogeremos la antigüedad máxima deseada, y lo entrenaremos con datos anteriores.
# POR TANTO, LOS RESULTADOS PUEDEN DIFERIR DE LOS QUE SE USEN EN EL FUTURO CON DINERO REAL.
# Por otro lado, SIEMPRE SE DEBEN ACTIVAR LAS DESCARGAS (en parametros.config) para que, en cada 
# antigüedad analizada, se obtenga la fila de futuro a predecir con la fecha deseada.
# ¡¡¡ATENCIÓN: AL ACTIVAR LAS DESCARGAS, SE BORRARÁN LOS MODELOS ACTUALES!!!! GUARDARLOS PREVIAMENTE
# Es OBLIGATORIO que en la carpeta usada haya un fichero GRANDE con antiguedad 0 en su nombre, y 
# al menos X días más reciente que el rango a analizar.

#Instantes de las descargas
#Se analizará el tramo de antiguedad desde máxima hasta minima
#Se tomarán los ficheros *_GRANDE_0_SG_0_* generados, o que ya se tienen de ejecuciones antiguas, para usarlo como base de información futura.
ANTIGUEDAD_MAXIMA="150"
ANTIGUEDAD_MINIMA="0" # Se puede usar cualquier valor
NUM_EMPRESAS_TRAIN="1000" # Número de empresas de entrenamiento, NO para los días posteriores (que estarán en el fichero de parámetros). Se recomienda dejar 1000 siempre.

echo -e "INVERSION - INICIO: "$( date "+%Y%m%d%H%M%S" )

#echo -e "Parando cron..."
#sudo service cron  stop

#################### DIRECTORIOS ###############################################################
DIR_CODIGOS_CARLOS="/home/carloslinux/Desktop/GIT_BOLSA/bolsa/"
DIR_CODIGOS_LUIS="/home/t151521/bolsa/"
PYTHON_MOTOR_CARLOS="/home/carloslinux/anaconda3/envs/BolsaPython38/bin/python"
PYTHON_MOTOR_LUIS="/home/t151521/anaconda3/envs/BolsaPython/bin/python"
DIR_DROPBOX_CARLOS="/home/carloslinux/Dropbox/BOLSA_PREDICTOR/"
DIR_DROPBOX_LUIS="/home/t151521/Dropbox/BOLSA_PREDICTOR/"
PATH_ANALISIS_LUIS="/home/t151521/bolsa/BolsaScripts/inversionAnalisis.sh"
PATH_ANALISIS_CARLOS="/home/carloslinux/Desktop/GIT_BOLSA/bolsa/BolsaScripts/inversionAnalisis.sh"

usuario=$(whoami)
if [ $usuario == "carloslinux" ]
then
  DIR_CODIGOS="${DIR_CODIGOS_CARLOS}"
  PYTHON_MOTOR="${PYTHON_MOTOR_CARLOS}"
  DIR_DROPBOX="${DIR_DROPBOX_CARLOS}"
  PATH_ANALISIS="${PATH_ANALISIS_CARLOS}"
elif [ $usuario == "t151521" ]
then
  DIR_CODIGOS="${DIR_CODIGOS_LUIS}"
  PYTHON_MOTOR="${PYTHON_MOTOR_LUIS}"
  DIR_DROPBOX="${DIR_DROPBOX_LUIS}"
  PATH_ANALISIS="${PATH_ANALISIS_LUIS}"
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
LOG_INVERSION="${DIR_LOGS}inversionHistoricaSoloFuturoSimplicado.log"
DIR_FUT_SUBGRUPOS="${DIR_BASE}futuro/subgrupos/"
DIR_JAVA="${DIR_CODIGOS}BolsaJava/"
PATH_JAR="${DIR_JAVA}target/bolsajava-1.0-jar-with-dependencies.jar"
DIR_GITHUB_INVERSION="${DIR_CODIGOS}inversion/"


#################################### CÓDIGO ###########################################################
DESCARGAR="INVALIDO"
for (( ANTIGUEDAD=${ANTIGUEDAD_MINIMA}; ANTIGUEDAD<=${ANTIGUEDAD_MAXIMA}; ANTIGUEDAD++ ))
do  

	

######## Para la antigüedad mínima (normalmente=0), se vacía la carpeta de futuro y se activan las descargas, pero para el resto no (porque se reutilizarán)
	if [ ${ANTIGUEDAD} -eq ${ANTIGUEDAD_MINIMA} ]; then

        	DESCARGAR="S"

		rm -Rf ${DIR_BASE}futuro/ >>${LOG_INVERSION}
		crearCarpetaSiNoExiste "${DIR_FUT_SUBGRUPOS}"
		crearCarpetaSiNoExiste "${DIR_FUT_SUBGRUPOS}brutos/"
		crearCarpetaSiNoExiste "${DIR_FUT_SUBGRUPOS}brutos_csv/"
		crearCarpetaSiNoExiste "${DIR_FUT_SUBGRUPOS}limpios/"
		crearCarpetaSiNoExiste "${DIR_FUT_SUBGRUPOS}elaborados/"
	
	fi

######## Para antigüedades mayores que la mínima, se asume que el fichero COMPLETO.csv de la antigüedad mínima ya existe, así que no se volverá a descargar, sino que se borrarán los días posteriores a esa antigüedad mínima (en la práctica se eliminan todas las filas de un día, para cada iteración), y se predecirá.
# El fichero resultante de la predicción sobreescribirá al anterior, porque no tiene fecha en su nombre, sino sólo variará su contenido.
# Antes que nada, se eliminan pasadas predicciones: ficheros *COMPLETO_PREDICCION*
	if [ ${ANTIGUEDAD} -gt ${ANTIGUEDAD_MINIMA} ]; then
                # Se evita descargar COMPLETO.csv
		DESCARGAR="N"

                # Se eliminan ficheros de anitguas predicciones (en TODOS los subgrupos)
                while IFS= read -r -d '' -u 9
		do
			if [[ $REPLY == *"COMPLETO_PREDICCION"* ]]; then
				echo "Se elimina el fichero  ${REPLY}  ..."  >>${LOG_INVERSION}
				ficheronombre=$(basename $REPLY)
				directorio=$(dirname $REPLY)
				rm "${directorio}/${ficheronombre}" >> ${LOG_INVERSION}
			fi
		done 9< <( find ${DIR_FUT_SUBGRUPOS} -type f -exec printf '%s\0' {} + )

                # Se elimina el siguiente día de COMPLETO.csv (en TODOS los subgrupos)
		while IFS= read -r -d '' -u 9
		do
			if [[ $REPLY == *"COMPLETO.csv" ]]; then
				echo "Procesamos  ${REPLY}  para eliminar todas las filas del día más reciente ..."  >>${LOG_INVERSION}
				ficheronombre=$(basename $REPLY)
				directorio=$(dirname $REPLY)
				$PYTHON_MOTOR "${PYTHON_SCRIPTS}bolsa/QuitarDiasMasRecientes.py" "${directorio}/${ficheronombre}" "1" "${directorio}/" "${ficheronombre}" >> ${LOG_INVERSION}
			fi
		done 9< <( find ${DIR_FUT_SUBGRUPOS} -type f -exec printf '%s\0' {} + )


	fi

	echo -e $( date '+%Y%m%d_%H%M%S' )" Ejecución del futuro (para velas de antiguedad=${ANTIGUEDAD}) con TODAS LAS EMPRESAS (lista DIRECTA ó INVERSA, ya da igual, no estamos mirando overfitting)..." >>${LOG_INVERSION}
	MIN_COBERTURA_CLUSTER=0    # Para predecir, cojo lo que haya, sin mínimos. El modelo ya lo hemos entrenado
	MIN_EMPRESAS_POR_CLUSTER=1   # Para predecir, cojo lo que haya, sin mínimos. El modelo ya lo hemos entrenado

        if [ ${ANTIGUEDAD} -eq ${ANTIGUEDAD_MINIMA} ]; then
	        ${PATH_SCRIPTS}masterh2o.sh "futuro" "${ANTIGUEDAD}" "0" "${DESCARGAR}" "S" "${S}" "${X}" "${R}" "${M}" "${F}" "${B}" "${NUM_EMPRESAS_INVERSION}" "${UMBRAL_SUBIDA_POR_VELA}" "${UMBRAL_MINIMO_GRAN_VELA}" "${MIN_COBERTURA_CLUSTER}" "${MIN_EMPRESAS_POR_CLUSTER}" "20001111" "20991111" "${MAX_NUM_FEAT_REDUCIDAS}" "${CAPA5_MAX_FILAS_ENTRADA}" "${DINAMICA1}" "${DINAMICA2}" 2>>${LOG_INVERSION} 1>>${LOG_INVERSION}
        fi

        if [ ${ANTIGUEDAD} -gt ${ANTIGUEDAD_MINIMA} ]; then
        	${PATH_SCRIPTS}masterSimplificadoh2o.sh "futuro" "${ANTIGUEDAD}" "0" "${DESCARGAR}" "S" "${S}" "${X}" "${R}" "${M}" "${F}" "${B}" "${NUM_EMPRESAS_INVERSION}" "${UMBRAL_SUBIDA_POR_VELA}" "${UMBRAL_MINIMO_GRAN_VELA}" "${MIN_COBERTURA_CLUSTER}" "${MIN_EMPRESAS_POR_CLUSTER}" "20001111" "20991111" "${MAX_NUM_FEAT_REDUCIDAS}" "${CAPA5_MAX_FILAS_ENTRADA}" "${DINAMICA1}" "${DINAMICA2}" 2>>${LOG_INVERSION} 1>>${LOG_INVERSION}
        fi
	
	while IFS= read -r -d '' -u 9
	do
		if [[ $REPLY == *"COMPLETO_PREDICCION"* ]]; then
			echo "Procesamos  ${REPLY}  y lo copiamos en ${DIR_DROPBOX} ..."  >>${LOG_INVERSION}
			ficheronombre=$(basename $REPLY)
			directorio=$(dirname $REPLY)
			$PYTHON_MOTOR "${PYTHON_SCRIPTS}bolsa/InversionUtils.py" "${directorio}/${ficheronombre}" "0" "${DIR_DROPBOX}" "${ficheronombre}" >> ${LOG_INVERSION}
		fi
	done 9< <( find ${DIR_FUT_SUBGRUPOS} -type f -exec printf '%s\0' {} + )



done


echo -e "INVERSION - FIN: "$( date "+%Y%m%d%H%M%S" )

