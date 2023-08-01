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



##############################################################################################################################
echo -e "$( date "+%Y%m%d%H%M%S" ) [inversion.sh] Se crea carpeta diaria de entregables y se meten dentro..." >>${LOG_INVERSION}
cd "${DIR_DROPBOX}"
DIR_ENTREGABLES_DROPBOX="${DIR_DROPBOX}"$(date '+%Y%m%d_%H%M%S')"/"
echo "DIR_ENTREGABLES_DROPBOX=${DIR_ENTREGABLES_DROPBOX}" >> ${LOG_INVERSION}
mkdir -p "${DIR_ENTREGABLES_DROPBOX}"
PREFIJO_RECIEN_CALCULADOS=$(ls -t | grep GRANDE | grep SG_0_ | head -n 1 | awk -F "_" '{print $1}')  # no tiene que cogerse AAAAMMDD de ahora, sino de los CSV mas recientes encontrados
echo "PREFIJO_RECIEN_CALCULADOS=${PREFIJO_RECIEN_CALCULADOS}" >> ${LOG_INVERSION}

if [ -z "${PREFIJO_RECIEN_CALCULADOS}" ]; then
    echo "Variable PREFIJO_RECIEN_CALCULADOS esta vacia!!" >>${LOG_INVERSION}
else
    echo "Copiando a carpeta entregables en DROPBOX..." >>${LOG_INVERSION}
    cp "${DIR_DROPBOX}/${PREFIJO_RECIEN_CALCULADOS}.html" "${DIR_ENTREGABLES_DROPBOX}/"
    cp "${DIR_DROPBOX}/${PREFIJO_RECIEN_CALCULADOS}_todas_las_empresas.html" "${DIR_ENTREGABLES_DROPBOX}/"
    cp "/bolsa/logs/pasado_metricas_y_rentabilidades.html" "${DIR_ENTREGABLES_DROPBOX}/"
    cp "/bolsa/pasado/empresas_clustering_web.html" "${DIR_ENTREGABLES_DROPBOX}/"

    ################################## GITHUB: commit and push##############################################################
    echo -e "$( date "+%Y%m%d%H%M%S" ) [inversion.sh] Copiando a carpeta entregables en GIT..." >>${LOG_INVERSION}
    DIR_DOCS_HTML_GIT="${DIR_CODIGOS}docs/${PREFIJO_RECIEN_CALCULADOS}/"
	echo "Copiando todos los entregables:  ${DIR_ENTREGABLES_DROPBOX} --> ${DIR_DOCS_HTML_GIT}" >>${LOG_INVERSION}
    mkdir -p "${DIR_DOCS_HTML_GIT}"
    cp ${DIR_ENTREGABLES_DROPBOX}/* ${DIR_DOCS_HTML_GIT}
    
	cd "${DIR_DOCS_HTML_GIT}"
    git add "."
    git commit -am "HTMLs del futuro (ID ejecucion: ${PREFIJO_RECIEN_CALCULADOS})"
    git push
    ################################################################################################
fi

echo -e "INVERSION - FIN: "$( date "+%Y%m%d%H%M%S" )

