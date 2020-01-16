#!/bin/bash

################## FUNCIONES #############################################################
crearCarpetaSiNoExiste() {
	param1=${1} 			#directorio
	echo "Creando carpeta: $param1"
	mkdir -p ${param1}
}

################################################################################################
VELAS_RETROCESO="0" #INSTANTE ANALIZADO, el ACTUAL

# PARAMETROS DE TARGET MEDIDOS EN VELAS
S="15"  #Subida durante [t1,t2]
X="28"  #Duracion en velas de [t1,t2]
R="12"  #Caida ligera máxima permitida durante [t2,t3], en TODAS esas velas.
M="7"  #Duración en velas de [t2,t3]
F="5"  #Caida ligera permitida durante [t2,t3], en la ÚLTIMA vela

DIR_BASE="/bolsa/"

DIR_CODIGOS_CARLOS="/home/carloslinux/Desktop/GIT_BOLSA/" #NO TOCAR
DIR_CODIGOS_LUIS="/home/t151521/bolsa/" #NO TOCAR
usuario=$(whoami)
if [ $usuario == "carloslinux" ]
then
  DIR_CODIGOS="${DIR_CODIGOS_CARLOS}"
elif [ $usuario == "t151521" ]
then
  DIR_CODIGOS="${DIR_CODIGOS_LUIS}"
else
  echo "ERROR: USUARIO NO CONTROLADO"
  s1 -e
fi


PATH_SCRIPTS="${DIR_CODIGOS}BolsaScripts/"
DIR_LOGS="${DIR_BASE}logs/"
LOG_PREDICTOR="${DIR_LOGS}predictor.log"
DIR_VALIDACION="${DIR_BASE}validacion/"
DIR_FUT_SUBGRUPOS="${DIR_BASE}futuro/subgrupos/"
DIR_JAVA="${DIR_CODIGOS}BolsaJava/"
PATH_JAR="${DIR_JAVA}target/bolsajava-1.0-jar-with-dependencies.jar"
DIR_PREDICCION="${DIR_BASE}prediccion/"


#rm -Rf ${DIR_LOGS}
#rm -Rf ${DIR_VALIDACION}
#crearCarpetaSiNoExiste "${DIR_LOGS}"
crearCarpetaSiNoExiste "${DIR_PREDICCION}"

############### LOGS ########################################################
#rm -f "${DIR_LOGS}log4j.log"
rm -f "${LOG_PREDICTOR}"

################################################################################################

echo -e "PARAMETROS --> VELAS_RETROCESO|S|X|R|M|F --> ${VELAS_RETROCESO}|${S}|${X}|${R}|${M}|${F}" >>${LOG_PREDICTOR}

################################################################################################

#En esta ejecucion, nos situamos unas velas atras y PREDECIMOS el futuro. Guardaremos esa predicción y la comparamos con lo que ha pasado hoy (dato REAL)

rm -Rf /bolsa/futuro/ >>${LOG_PREDICTOR}

echo -e "Ejecución del futuro (para velas de antiguedad=${VELAS_RETROCESO}) con OTRAS empresas (lista REVERTIDA)..." >>${LOG_PREDICTOR}
${PATH_SCRIPTS}master.sh "futuro" "$VELAS_RETROCESO" "1" "S" "${S}" "${X}" "${R}" "${M}" "${F}"  2>>${LOG_PREDICTOR} 1>>${LOG_PREDICTOR}



dir_val_logs="/bolsa/validacion/E600_VR${VELAS_RETROCESO}_S${S}_X${X}_R${R}_M${M}_F${F}_log/"
rm -Rf $dir_val_logs
mkdir -p $dir_val_logs
cp -R "/bolsa/logs/" $dir_val_logs

echo -e "******** FIN de validacion **************" >> ${LOG_PREDICTOR}




