#!/bin/bash

################## FUNCIONES #############################################################
crearCarpetaSiNoExiste() {
	param1=${1} 			#directorio
	echo "Creando carpeta: $param1"
	mkdir -p ${param1}
}

################################################################################################
DIR_TIEMPO="pasado"

DIR_BASE="/bolsa/"
DIR_LOGS="${DIR_BASE}logs/"
DIR_BRUTOS="${DIR_BASE}${DIR_TIEMPO}/brutos/"
DIR_BRUTOS_CSV="${DIR_BASE}${DIR_TIEMPO}/brutos_csv/"
DIR_LIMPIOS="${DIR_BASE}${DIR_TIEMPO}/limpios/"
DIR_ELABORADOS="${DIR_BASE}${DIR_TIEMPO}/elaborados/"
DIR_SUBGRUPOS="${DIR_BASE}${DIR_TIEMPO}/subgrupos/"
DIR_IMG="img/"
DIR_INTEGRACION="${DIR_BASE}integracion/"

crearCarpetaSiNoExiste "${DIR_INTEGRACION}"
cd ${DIR_INTEGRACION}

############### LOGS ########################################################
LOG_INTEGRACION="${DIR_LOGS}integracion.log"
rm -f "${LOG_INTEGRACION}"

################################################################################################
echo -e ""  >${LOG_INTEGRACION}
mkdir -p $DIR_LOGS >>${LOG_INTEGRACION}

echo -e "******** INICIO del test de integracion **************" >> ${LOG_INTEGRACION}

echo -e "Nos saltamos la capa de DESCARGA (ponemos unos datos que tenemos ya guardados de otro día)..." >> ${LOG_INTEGRACION}
rm -Rf /bolsa/pasado/ >>${LOG_INTEGRACION}
mkdir -p "${DIR_BRUTOS_CSV}" >>${LOG_INTEGRACION}

cp ${DIR_INTEGRACION}brutos_csv/* "${DIR_BRUTOS_CSV}"  2>>${LOG_INTEGRACION} 1>>${LOG_INTEGRACION}

echo -e "Ejecutando MASTER (saltando la capa de descarga)..." >> ${LOG_INTEGRACION}
${PATH_SCRIPTS}master.sh "pasado" "0" "0" "N"  2>>${LOG_INTEGRACION} 1>>${LOG_INTEGRACION}

################################################################################################
echo -e "****Comprobaciones de INTEGRACION*********" >> ${LOG_INTEGRACION}

NUM_FICHEROS_10x=$(ls -l ${DIR_BRUTOS_CSV} | wc -l)
echo -e "La capa 10X ha generado $NUM_FICHEROS_10x ficheros" 2>>${ LOG_INTEGRACION} 1>>${ LOG_INTEGRACION}
if [ "$NUM_FICHEROS_10x" -lt 1 ]; then
	echo -e "El numero de ficheros es menor que el de la capa anterior. Asi que se han perdido algunos por algun problema. Debemos analizarlo. Saliendo..." 2>>${ LOG_INTEGRACION} 1>>${ LOG_INTEGRACION}
	exit -1
fi

NUM_FICHEROS_20x=$(ls -l ${DIR_LIMPIOS} | wc -l)
echo -e "La capa 20X ha generado $NUM_FICHEROS_20x ficheros" 2>>${ LOG_INTEGRACION} 1>>${ LOG_INTEGRACION}
if [ "$NUM_FICHEROS_20x" -lt "$NUM_FICHEROS_10x" ]; then
	echo -e "El numero de ficheros es menor que el de la capa anterior. Asi que se han perdido algunos por algun problema. Debemos analizarlo. Saliendo..." 2>>${ LOG_INTEGRACION} 1>>${ LOG_INTEGRACION}
	exit -1
fi

NUM_FICHEROS_30x=$(ls -l ${DIR_ELABORADOS} | wc -l)
echo -e "La capa 30X ha generado $NUM_FICHEROS_30x ficheros" 2>>${ LOG_INTEGRACION} 1>>${ LOG_INTEGRACION}
if [ "$NUM_FICHEROS_30x" -lt "$NUM_FICHEROS_20x" ]; then
	echo -e "El numero de ficheros es menor que el de la capa anterior. Asi que se han perdido algunos por algun problema. Debemos analizarlo. Saliendo..." 2>>${ LOG_INTEGRACION} 1>>${ LOG_INTEGRACION}
	exit -1
fi

echo -e "******** FIN del test de integracion **************" >> ${LOG_INTEGRACION}


