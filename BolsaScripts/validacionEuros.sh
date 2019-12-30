#!/bin/bash

################## FUNCIONES #############################################################
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

################################################################################################
VELAS_RETROCESO="50"

DIR_BASE="/bolsa/"
DIR_CODIGOS="/home/carloslinux/Desktop/GIT_BOLSA/"
PATH_SCRIPTS="${DIR_CODIGOS}BolsaScripts/"
DIR_LOGS="${DIR_BASE}logs/"
DIR_VALIDACION="${DIR_BASE}validacion/"
DIR_FUT_SUBGRUPOS="${DIR_BASE}futuro/subgrupos/"

crearCarpetaSiNoExisteYVaciar "${DIR_VALIDACION}"

############### LOGS ########################################################
rm -f "${DIR_LOGS}log4j.log"
LOG_VALIDADOR="${DIR_LOGS}${ID_EJECUCION}_bolsa_coordinador_${DIR_TIEMPO}.log"
rm -f "${LOG_VALIDADOR}"

############### COMPILAR JAR ########################################################
echo -e "Compilando JAVA en un JAR..." >> ${LOG_VALIDADOR}
cd "${DIR_JAVA}"
mvn clean compile assembly:single

################################################################################################

echo -e "Ejecución del pasado (para entrenar los modelos como si estuvieramos ATRAS con la lista normal de empresas)..." >>${LOG_VALIDADOR}
${PATH_SCRIPTS}master.sh "pasado" "$VELAS_RETROCESO" "0"  2>>${LOG_VALIDADOR} 1>>${LOG_VALIDADOR}

echo -e "Ejecución del futuro (para las velas que habia ATRAS con la lista normal de empresas)..." >>${LOG_VALIDADOR}
${PATH_SCRIPTS}master.sh "futuro" "$VELAS_RETROCESO" "1"   2>>${LOG_VALIDADOR} 1>>${LOG_VALIDADOR}

echo -e "ATRAS $VELAS_RETROCESO velas --> Guardamos la prediccion de todos los SUBGRUPOS en la carpeta de validacion, para analizarla luego..." >>${LOG_VALIDADOR}
while IFS= read -r -d '' -u 9
do
	if [[ $REPLY == *"COMPLETO_PREDICCION"* ]]; then
		echo "Copiando este fichero   $REPLY   ..." >>${LOG_VALIDADOR}
		cp $REPLY $DIR_VALIDACION
	fi
done 9< <( find $DIR_FUT_SUBGRUPOS -type f -exec printf '%s\0' {} + )


echo -e "Ejecución del futuro (para velas de HOY) con OTRAS empresas (lista revertida)..." >>${LOG_VALIDADOR}
${PATH_SCRIPTS}master.sh "futuro" "0" "1"   2>>${LOG_VALIDADOR} 1>>${LOG_VALIDADOR}

echo -e "VELAS_50 --> Guardamos la prediccion de todos los SUBGRUPOS en la carpeta de validacion, para analizarla luego..." >>${LOG_VALIDADOR}
while IFS= read -r -d '' -u 9
do
	if [[ $REPLY == *"COMPLETO_PREDICCION"* ]]; then
		echo "Copiando este fichero   $REPLY   ..." >>${LOG_VALIDADOR}
		cp $REPLY $DIR_VALIDACION
	fi
done 9< <( find $DIR_FUT_SUBGRUPOS -type f -exec printf '%s\0' {} + )

echo -e "Validacion de rendimiento..." >> ${LOG_VALIDADOR}
java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c70X.validacion.Validador" "${PATH_PREDICHO}" "${PATH_VALIDACION}" "${S}" "${X}" "${R}" "${M}" 2>>${LOG_VALIDADOR} 1>>${LOG_VALIDADOR}


