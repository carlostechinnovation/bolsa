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


echo -e "$( date "+%Y%m%d%H%M%S" ) [inversion.sh] Se crea carpeta diaria de entregables y se meten dentro..." >>${LOG_INVERSION}
cd "${DIR_DROPBOX}"
CARPETA_ENTREGABLES="${DIR_DROPBOX}"$(date '+%Y%m%d_%H%M%S')
echo "CARPETA_ENTREGABLES=${CARPETA_ENTREGABLES}" >> ${LOG_INVERSION}
mkdir -p "${CARPETA_ENTREGABLES}"
PREFIJO_RECIEN_CALCULADOS=$(ls -t | grep GRANDE | grep SG_0_ | head -n 1 | awk -F "_" '{print $1}')  # no tiene que cogerse AAAAMMDD de ahora, sino de los CSV mas recientes encontrados
echo "PREFIJO_RECIEN_CALCULADOS=${PREFIJO_RECIEN_CALCULADOS}" >> ${LOG_INVERSION}

if [ -z "${PREFIJO_RECIEN_CALCULADOS}" ]; then
    echo "PREFIJO_RECIEN_CALCULADOS esta vacio!!" >>${LOG_INVERSION}
else
    echo "Copiando a carpeta entregables..." >>${LOG_INVERSION}
    cp "${DIR_DROPBOX}/${PREFIJO_RECIEN_CALCULADOS}.html" "${CARPETA_ENTREGABLES}/"
    cp "${DIR_DROPBOX}/${PREFIJO_RECIEN_CALCULADOS}_todas_las_empresas.html" "${CARPETA_ENTREGABLES}/"
    cp "/bolsa/logs/pasado_metricas_y_rentabilidades.html" "${CARPETA_ENTREGABLES}/"
    cp "/bolsa/pasado/empresas_clustering_web.html" "${CARPETA_ENTREGABLES}/"

    ################################## GITHUB: commit and push##############################################################
    echo -e "$( date "+%Y%m%d%H%M%S" ) [inversion.sh] Haciendo GIT COMMIT..." >>${LOG_INVERSION}
    #Subir HTML resultado a GIT
    DIR_DOCS_HTML_GIT="${DIR_CODIGOS}docs/PREFIJO_RECIEN_CALCULADOS/"
    mkdir -p "${DIR_DOCS_HTML_GIT}"
    cp "${CARPETA_ENTREGABLES}/*.html" "${DIR_DOCS_HTML_GIT}"
    
    cd "${DIR_DOCS_HTML_GIT}"
    git add "."
    git commit -am "HTMLs del futuro (ID ejecucion: ${PREFIJO_RECIEN_CALCULADOS})"
    git push
    ################################################################################################
fi

echo -e "INVERSION - FIN: "$( date "+%Y%m%d%H%M%S" )

