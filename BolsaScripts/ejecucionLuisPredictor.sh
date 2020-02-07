#!/bin/bash

################## FUNCIONES #############################################################
crearCarpetaSiNoExiste() {
	param1=${1} 			#directorio
	echo "Creando carpeta: $param1"
	mkdir -p ${param1}
}

################################################################################################
# PARAMETROS DE TARGET MEDIDOS EN VELAS
S="15"  #Subida durante [t1,t2]
X="4"  #Duracion en velas de [t1,t2]
R="5"  #Caida ligera máxima permitida durante [t2,t3], en TODAS esas velas.
M="1"  #Duración en velas de [t2,t3]
F="5"  #Caida ligera permitida durante [t2,t3], en la ÚLTIMA vela
B="10"  #Caida ligera permitida durante [t1,t2], en todas esas velas
NUM_EMPRESAS="50"  #Numero de empresas descargadas
ACTIVAR_DESCARGAS="-" #Descargar datos nuevos (S) o usar datos locales (N)
UMBRAL_SUBIDA_POR_VELA="3" #Recomendable: 3. Umbral de subida máxima relativa de una vela respecto de subida media, en velas de 1 a X. 

VELAS_RETROCESO="$((${X}+${M}+2))" #INSTANTE ANALIZADO (T1). Su antiguedad debe ser mayor que X+M, para poder ver esas X+M velas del futuro


#Instantes de las descargas
PASADO_t1="0"
FUTURO1_t1="${VELAS_RETROCESO}"
FUTURO2_t1="$((${VELAS_RETROCESO}-${X}-${M}))"

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

################################################################################################
# Prediccion del FUTURO para poner dinero. Esto va a crontab
################################################################################################

LOG_REAL="/bolsa/logs/predictor.log"
rm -Rf /bolsa/futuro/ >>${LOG_REAL}
crearCarpetaSiNoExiste "${DIR_FUT_SUBGRUPOS}"
AHORA="0"
NUM_EMPRESAS_REAL="50"
DIR_REAL="${DIR_BASE}real/"

crearCarpetaSiNoExiste "${DIR_REAL}"

echo -e $( date '+%Y%m%d_%H%M%S' )" Ejecución del futuro (para velas de antiguedad=${AHORA}) con empresas donde poner dinero REAL..." >>${LOG_REAL}
${PATH_SCRIPTS}master.sh "futuro" "${AHORA}" "1" "S" "S" "${S}" "${X}" "${R}" "${M}" "${F}" "${B}" "${NUM_EMPRESAS_REAL}" "${UMBRAL_SUBIDA_POR_VELA}" 2>>${LOG_REAL} 1>>${LOG_REAL}

echo -e "Velas con antiguedad=AHORA --> Guardamos la prediccion de todos los SUBGRUPOS en la carpeta de validacion, para analizarla luego..." >>${LOG_REAL}
while IFS= read -r -d '' -u 9
do
	if [[ $REPLY == *"COMPLETO_PREDICCION"* ]]; then
		echo "Copiando este fichero   $REPLY   ..." >>${LOG_REAL}
		cp $REPLY $DIR_REAL
	fi
done 9< <( find $DIR_FUT_SUBGRUPOS -type f -exec printf '%s\0' {} + )


echo -e "" > "${DIR_REAL}/caracteristicas.log"
echo -e "Instante de ejecución: "$( date '+%Y%m%d_%H%M%S' )"\n\n" >>${LOG_VALIDADOR}
echo -e "\n------------------------ COBERTURA --------------------------" >> "${DIR_REAL}/caracteristicas.log"
cat "/bolsa/logs/validador.log" | grep 'COBERTURA' >> "${DIR_REAL}/caracteristicas.log"
echo -e "\n------------------------ RENTABILIDAD --------------------------" >> "${DIR_REAL}/caracteristicas.log"
cat "/bolsa/logs/validador.log" | grep 'RENTABILIDAD' >> "${DIR_REAL}/caracteristicas.log"
echo -e "\n----------------------------------------------------------------------------" >> "${DIR_REAL}/caracteristicas.log"
################################################################################################

