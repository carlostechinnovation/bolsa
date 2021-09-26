#!/bin/bash

echo -e "VALIDACION - INICIO: "$( date "+%Y%m%d%H%M%S" )

#################### DIRECTORIOS ###############################################################
DIR_BASE="/bolsa/"
DIR_CODIGOS_CARLOS="/home/carloslinux/Desktop/GIT_BOLSA/"
DIR_CODIGOS_LUIS="/home/t151521${DIR_BASE}"
PYTHON_MOTOR_CARLOS="/home/carloslinux/anaconda3/envs/BolsaPython38/bin/python"
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

VELAS_RETROCESO="$((${X}+${M}+2))" #INSTANTE ANALIZADO (T1). Su antiguedad debe ser mayor o igual que X+M, para poder ver esas X+M velas del futuro

#Instantes de las descargas
PASADO_t1="50"
FUTURO1_t1="${VELAS_RETROCESO}" #No tocar. Se usa en el validador
FUTURO2_t1="$((${VELAS_RETROCESO}-${X}-${M}))" #No tocar. Se usa en el validador

DIR_LOGS="${DIR_BASE}logs/"
LOG_VALIDADOR="${DIR_LOGS}validador.log"
DIR_VALIDACION="${DIR_BASE}validacion/"
DIR_FUT_SUBGRUPOS="${DIR_BASE}futuro/subgrupos/"
DIR_JAVA="${DIR_CODIGOS}BolsaJava/"
PATH_JAR="${DIR_JAVA}target/bolsajava-1.0-jar-with-dependencies.jar"


rm -Rf ${DIR_LOGS}
rm -Rf ${DIR_VALIDACION}
crearCarpetaSiNoExiste "${DIR_LOGS}"
crearCarpetaSiNoExiste "${DIR_VALIDACION}"

############### LOGS ########################################################
rm -f "${DIR_LOGS}log4j.log"
rm -f "${LOG_VALIDADOR}"

################################################################################################

echo -e "PARAMETROS --> VELAS_RETROCESO|S|X|R|M|F|B|NUM_EMPRESAS|SUBIDA_MAXIMA_POR_VELA|MIN_COBERTURA_CLUSTER|MIN_EMPRESAS_POR_CLUSTER|MAX_NUM_FEAT_REDUCIDAS|RANGO_YF|VELA_YF --> ${VELAS_RETROCESO}|${S}|${X}|${R}|${M}|${F}|${B}|${NUM_EMPRESAS}|${UMBRAL_SUBIDA_POR_VELA}|${MIN_COBERTURA_CLUSTER}|${MIN_EMPRESAS_POR_CLUSTER}|${MAX_NUM_FEAT_REDUCIDAS}|${RANGO_YF}|${VELA_YF}|${DINAMICA1}|${DINAMICA2}" >>${LOG_VALIDADOR}
echo -e "PARAMETROS -->  PASADO_t1 | FUTURO1_t1 | FUTURO2_t1--> ${PASADO_t1}|${FUTURO1_t1}|${FUTURO2_t1}" >>${LOG_VALIDADOR}

rm -Rf ${DIR_BASE}pasado/ >>${LOG_VALIDADOR}
mkdir -p ${DIR_BASE}logs/ >>${LOG_VALIDADOR}
echo -e ""  >>${LOG_VALIDADOR}

if [ "${ACTIVAR_DESCARGAS}" = "N" ];  then
	echo -e "PASADO - Usamos datos LOCALES (sin Internet) de la ruta ${DIR_BASE}datos_validacion/" >>${LOG_VALIDADOR}
	crearCarpetaSiNoExiste "${DIR_BASE}pasado/brutos_csv/"
	cp -a "${DIR_BASE}validacion_datos/pasado_brutos_csv/." "${DIR_BASE}pasado/brutos_csv/" >>${LOG_VALIDADOR}
fi;

echo -e $( date '+%Y%m%d_%H%M%S' )" Ejecución del PASADO (para entrenar los modelos a dia de HOY con la lista normal de empresas)..." >>${LOG_VALIDADOR}

echo -e "EL MODELO SE ESTÁ GENERANDO EN EL PASADO DE HACE ${PASADO_t1} VELAS." 
echo -e "EL MODELO SE ESTÁ GENERANDO EN EL PASADO DE HACE ${PASADO_t1} VELAS." >>${LOG_VALIDADOR}


${PATH_SCRIPTS}master.sh "pasado" "${PASADO_t1}" "0" "${ACTIVAR_DESCARGAS}" "S" "${S}" "${X}" "${R}" "${M}" "${F}" "${B}" "${NUM_EMPRESAS}" "${UMBRAL_SUBIDA_POR_VELA}" "${UMBRAL_MINIMO_GRAN_VELA}" "${MIN_COBERTURA_CLUSTER}" "${MIN_EMPRESAS_POR_CLUSTER}" "${P_INICIO}" "${P_FIN}" "${MAX_NUM_FEAT_REDUCIDAS}" "${CAPA5_MAX_FILAS_ENTRADA}" "${DINAMICA1}" "${DINAMICA2}" 2>>${LOG_VALIDADOR} 1>>${LOG_VALIDADOR}

dir_val_pasado="${DIR_BASE}validacion/E${NUM_EMPRESAS}_VR${VELAS_RETROCESO}_S${S}_X${X}_R${R}_M${M}_F${F}_B${B}_pasado/"
rm -Rf $dir_val_pasado
mkdir -p $dir_val_pasado
cp -R "${DIR_BASE}pasado/" $dir_val_pasado

echo -e "GENERA MODELO - FIN: "$( date "+%Y%m%d%H%M%S" )


