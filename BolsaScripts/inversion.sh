#!/bin/bash

#set -e

echo -e "INVERSION - INICIO: "$( date "+%Y%m%d%H%M%S" )

#################### DIRECTORIOS ###############################################################
DIR_CODIGOS_CARLOS="/home/carloslinux/Desktop/GIT_BOLSA/bolsa/"
DIR_CODIGOS_LUIS="/home/t151521/bolsa/"
PYTHON_MOTOR_CARLOS="/home/carloslinux/anaconda3/envs/BolsaPython38/bin/python"
PYTHON_MOTOR_LUIS="/usr/bin/python3.8"
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

#Instantes de las descargas
FUTURO_INVERSION="0" #Ahora mismo

DIR_BASE="/bolsa/"
DIR_LOGS="${DIR_BASE}logs/"
LOG_INVERSION="${DIR_LOGS}inversion.log"
DIR_INVERSION="${DIR_BASE}inversion/"
DIR_FUT_SUBGRUPOS="${DIR_BASE}futuro/subgrupos/"
DIR_JAVA="${DIR_CODIGOS}BolsaJava/"
PATH_JAR="${DIR_JAVA}target/bolsajava-1.0-jar-with-dependencies.jar"


rm -Rf ${DIR_INVERSION}
crearCarpetaSiNoExiste "${DIR_INVERSION}"

############### LOGS ########################################################
rm -f "${DIR_LOGS}log4j.log"
rm -f "${LOG_INVERSION}"

############### COMPILAR JAR ########################################################
echo -e "Comprobando que JAVA tenga su JAR aunque no lo usemos en este script..." >> ${LOG_INVERSION}

if [ -f "$PATH_JAR" ]; then
    echo "El siguiente JAR se ha generado bien: ${PATH_JAR}"
else 
    echo "El siguiente JAR no se ha generado bien: ${PATH_JAR}   Saliendo..."
	exit -1
fi

################################################################################################

#En esta ejecucion, nos situamos en HOY MISMO y PREDECIMOS el futuro. Guardaremos esa predicción para meter dinero REAL.

rm -Rf /bolsa/futuro/ >>${LOG_INVERSION}
crearCarpetaSiNoExiste "${DIR_FUT_SUBGRUPOS}"

echo -e $( date '+%Y%m%d_%H%M%S' )" Ejecución del futuro (para velas de antiguedad=0) con TODAS LAS EMPRESAS (lista DIRECTA ó INVERSA, ya da igual, no estamos mirando overfitting)..." >>${LOG_INVERSION}
MIN_COBERTURA_CLUSTER=0    # Para predecir, cojo lo que haya, sin minimos. EL modelo ya lo hemos entrenado
MIN_EMPRESAS_POR_CLUSTER=1   # Para predecir, cojo lo que haya, sin minimos. EL modelo ya lo hemos entrenado
${PATH_SCRIPTS}master.sh "futuro" "${FUTURO_INVERSION}" "0" "S" "S" "${S}" "${X}" "${R}" "${M}" "${F}" "${B}" "${NUM_EMPRESAS_INVERSION}" "${UMBRAL_SUBIDA_POR_VELA}" "${UMBRAL_MINIMO_GRAN_VELA}" "${MIN_COBERTURA_CLUSTER}" "${MIN_EMPRESAS_POR_CLUSTER}" "20001111" "20991111" "${MAX_NUM_FEAT_REDUCIDAS}" "${CAPA5_MAX_FILAS_ENTRADA}" "${DINAMICA1}" "${DINAMICA2}" 2>>${LOG_INVERSION} 1>>${LOG_INVERSION}

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

# En la carpeta DROPBOX, coge el CSV más reciente de predicciones (su nombre es 202XMMDD) y crea un fichero llamado 202XMMDD.html con toda la info que encuentre de ese dia. Ademas, le añade la info de CALIDAD.csv que esta aparte
$PYTHON_MOTOR "${PYTHON_SCRIPTS}bolsa/InversionJuntarPrediccionesDeUnDia.py" "${DIR_DROPBOX}" "${DIR_DROPBOX}/ANALISIS/CALIDAD.csv" "${DIR_CODIGOS}BolsaJava/src/main/resources/Bolsa_Subgrupos_Descripcion.txt" "${DIR_CODIGOS}BolsaJava/realimentacion/falsospositivos_empresas.csv" >> ${LOG_INVERSION}


# Crear CARPETA DIARIA DE ENTREGABLES y meterlos dentro
CARPETA_ENTREGABLES="${DIR_DROPBOX}/"$(date '+%Y%m%d_%H%M%S')
echo "CARPETA_ENTREGABLES=${CARPETA_ENTREGABLES}" >> ${LOG_INVERSION}
mkdir -p "${CARPETA_ENTREGABLES}"
PREFIJO_RECIEN_CALCULADOS=$(ls -t | grep GRANDE | grep SG_0_ | head -n 1 | awk -F "_" '{print $1}')  # no tiene que cogerse AAAAMMDD de ahora, sino de los CSV mas recientes encontrados
echo "PREFIJO_RECIEN_CALCULADOS=${PREFIJO_RECIEN_CALCULADOS}" >> ${LOG_INVERSION}
cp "${DIR_DROPBOX}/${PREFIJO_RECIEN_CALCULADOS}.html" "${CARPETA_ENTREGABLES}/"
cp "${DIR_DROPBOX}/${PREFIJO_RECIEN_CALCULADOS}_todas_las_empresas.html" "${CARPETA_ENTREGABLES}/"
cp "/bolsa/logs/pasado_metricas_y_rentabilidades.html" "${CARPETA_ENTREGABLES}/"
cp "/bolsa/pasado/empresas_clustering_web.html" "${CARPETA_ENTREGABLES}/"



################################## GITHUB: commit and push##############################################################
echo -e "Haciendo GIT COMMIT..." >>${LOG_INVERSION}
#Subir HTML resultado a GIT
DIR_DOCS_HTML_GIT="${DIR_CODIGOS}docs/PREFIJO_RECIEN_CALCULADOS/"
mkdir -p "${DIR_DOCS_HTML_GIT}"
cp "${CARPETA_ENTREGABLES}/*.html" "${DIR_DOCS_HTML_GIT}"

cd "${DIR_DOCS_HTML_GIT}"
git add "."
git commit -am "HTMLs del futuro (ID ejecucion: ${PREFIJO_RECIEN_CALCULADOS})"
git push

################################################################################################

echo -e "INVERSION - FIN: "$( date "+%Y%m%d%H%M%S" )

