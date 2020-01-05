#!/bin/bash

################## FUNCIONES #############################################################
crearCarpetaSiNoExiste() {
	param1=${1} 			#directorio
	echo "Creando carpeta: $param1"
	mkdir -p ${param1}
}

################################################################################################
VELAS_RETROCESO="50"

# PARAMETROS DE TARGET MEDIDOS EN VELAS
S="10"
X="56"
R="10"
M="7"
F="5"


DIR_BASE="/bolsa/"
DIR_CODIGOS="/home/carloslinux/Desktop/GIT_BOLSA/"
PATH_SCRIPTS="${DIR_CODIGOS}BolsaScripts/"
DIR_LOGS="${DIR_BASE}logs/"
LOG_VALIDADOR="${DIR_LOGS}validador.log"
DIR_VALIDACION="${DIR_BASE}validacion/"
DIR_FUT_SUBGRUPOS="${DIR_BASE}futuro/subgrupos/"
DIR_JAVA="${DIR_CODIGOS}BolsaJava/"
PATH_JAR="${DIR_JAVA}target/bolsajava-1.0-jar-with-dependencies.jar"


rm -Rf ${DIR_VALIDACION} >>${LOG_VALIDADOR}
crearCarpetaSiNoExiste "${DIR_VALIDACION}"

############### LOGS ########################################################
rm -f "${DIR_LOGS}log4j.log" >>${LOG_VALIDADOR}

rm -f "${LOG_VALIDADOR}" >>${LOG_VALIDADOR}

################################################################################################

rm -Rf /bolsa/pasado/ >>${LOG_VALIDADOR}
mkdir -p /bolsa/logs/ >>${LOG_VALIDADOR}
echo -e ""  >>${LOG_VALIDADOR}

echo -e "Ejecución del PASADO (para entrenar los modelos a dia de HOY con la lista normal de empresas)..." >>${LOG_VALIDADOR}
${PATH_SCRIPTS}master.sh "pasado" "0" "0" "S" "${S}" "${X}" "${R}" "${M}" "${F}" 2>>${LOG_VALIDADOR} 1>>${LOG_VALIDADOR}

################################################################################################
rm -Rf /bolsa/futuro/ >>${LOG_VALIDADOR}

echo -e "Ejecución del futuro (para velas de ATRAS) con OTRAS empresas (lista REVERTIDA)..." >>${LOG_VALIDADOR}
${PATH_SCRIPTS}master.sh "futuro" "$VELAS_RETROCESO" "1" "S" "${S}" "${X}" "${R}" "${M}" "${F}"  2>>${LOG_VALIDADOR} 1>>${LOG_VALIDADOR}

echo -e "ATRAS $VELAS_RETROCESO velas --> Guardamos la prediccion de todos los SUBGRUPOS en la carpeta de validacion, para analizarla luego..." >>${LOG_VALIDADOR}
while IFS= read -r -d '' -u 9
do
	if [[ $REPLY == *"COMPLETO_PREDICCION"* ]]; then
		echo "Copiando este fichero   $REPLY   ..." >>${LOG_VALIDADOR}
		cp $REPLY $DIR_VALIDACION
	fi
done 9< <( find $DIR_FUT_SUBGRUPOS -type f -exec printf '%s\0' {} + )

################################################################################################
rm -Rf /bolsa/futuro/ >>${LOG_VALIDADOR}

echo -e "Ejecución del futuro (para velas de HOY) con OTRAS empresas (lista REVERTIDA)..." >>${LOG_VALIDADOR}
${PATH_SCRIPTS}master.sh "futuro" "0" "1" "S" "${S}" "${X}" "${R}" "${M}" "${F}" 2>>${LOG_VALIDADOR} 1>>${LOG_VALIDADOR}

echo -e "VELAS_50 --> Guardamos la prediccion de todos los SUBGRUPOS en la carpeta de validacion, para analizarla luego..." >>${LOG_VALIDADOR}
while IFS= read -r -d '' -u 9
do
	if [[ $REPLY == *"COMPLETO_PREDICCION"* ]]; then
		echo "Copiando este fichero   $REPLY   ..." >>${LOG_VALIDADOR}
		cp $REPLY $DIR_VALIDACION
	fi
done 9< <( find $DIR_FUT_SUBGRUPOS -type f -exec printf '%s\0' {} + )

################################################################################################
echo -e "Validacion de rendimiento: comparar ATRAS_PREDICHO con HOY_REAL para empresas de la lista REVERTIDA..." >> ${LOG_VALIDADOR}
java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c70X.validacion.Validador" "${VELAS_RETROCESO}" "${DIR_VALIDACION}" "${S}" "${X}" "${R}" "${M}" 2>>${LOG_VALIDADOR} 1>>${LOG_VALIDADOR}

echo -e "Un sistema CLASIFICADOR BINOMIAL tonto acierta el 50% de las veces. El nuestro..." >> ${LOG_VALIDADOR}

echo -e "******** FIN de validacion **************" >> ${LOG_VALIDADOR}
