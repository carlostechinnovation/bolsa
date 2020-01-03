#!/bin/bash

################## FUNCIONES #############################################################
crearCarpetaSiNoExiste() {
	param1=${1} 			#directorio
	echo "Creando carpeta: $param1"
	mkdir -p ${param1}
}


################################################################################################
DIR_BASE="/bolsa/"
DIR_INTEGRACION="${DIR_BASE}integracion/"

rm -Rf ${DIR_INTEGRACION} >>${LOG_INTEGRACION}
crearCarpetaSiNoExiste "${DIR_INTEGRACION}"
cd ${DIR_INTEGRACION}

############### LOGS ########################################################
LOG_INTEGRACION="${DIR_LOGS}testIntegracion.log"
rm -f "${LOG_INTEGRACION}"

################################################################################################

rm -Rf /bolsa/pasado/ >>${LOG_INTEGRACION}
mkdir -p /bolsa/logs/ >>${LOG_INTEGRACION}
echo -e ""  >>${LOG_INTEGRACION}


echo -e "Nos saltamos la capa de DESCARGA (ponemos unos datos que tenemos ya guardados de otro día)..." >> ${LOG_INTEGRACION}
cp ${DIR_INTEGRACION}brutos_csv/* /bolsa/pasado/brutos_csv/  2>>${LOG_INTEGRACION} 1>>${LOG_INTEGRACION}
${PATH_SCRIPTS}master.sh "pasado" "0" "0" "N"  2>>${LOG_INTEGRACION} 1>>${LOG_INTEGRACION}

echo -e "******** FIN del test de integracion **************" >> ${LOG_INTEGRACION}


