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
ACTIVAR_DESCARGAS="S" #Descargar datos nuevos (S) o usar datos locales (N)
UMBRAL_SUBIDA_POR_VELA="3" #Recomendable: 3. Umbral de subida máxima relativa de una vela respecto de subida media, en velas de 1 a X. 

VELAS_RETROCESO=0

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

#En esta ejecucion, nos situamos unas velas atras y PREDECIMOS el futuro. Guardaremos esa predicción y la comparamos con lo que ha pasado hoy (dato REAL)

rm -Rf /bolsa/futuro/ >>${LOG_VALIDADOR}
crearCarpetaSiNoExiste "${DIR_FUT_SUBGRUPOS}"

if [ "${ACTIVAR_DESCARGAS}" = "N" ];  then
	echo -e "FUTURO1 - Usamos datos LOCALES (sin Internet) de la ruta /bolsa/datos_validacion/" >>${LOG_VALIDADOR}
	crearCarpetaSiNoExiste "/bolsa/futuro/brutos_csv/"
	cp -a "/bolsa/validacion_datos/futuro1_brutos_csv/." "/bolsa/futuro/brutos_csv/" >>${LOG_VALIDADOR}
fi;

echo -e $( date '+%Y%m%d_%H%M%S' )" Ejecución del futuro (para velas de antiguedad=${FUTURO1_t1}) con OTRAS empresas (lista REVERTIDA)..." >>${LOG_VALIDADOR}
${PATH_SCRIPTS}master.sh "futuro" "0" "1" "${ACTIVAR_DESCARGAS}" "S" "${S}" "${X}" "${R}" "${M}" "${F}" "${B}" "${NUM_EMPRESAS}" "${UMBRAL_SUBIDA_POR_VELA}" 2>>${LOG_VALIDADOR} 1>>${LOG_VALIDADOR}

echo -e "Atrás $VELAS_RETROCESO velas --> Guardamos la prediccion de todos los SUBGRUPOS en la carpeta de validacion, para analizarla luego..." >>${LOG_VALIDADOR}
while IFS= read -r -d '' -u 9
do
	if [[ $REPLY == *"COMPLETO_PREDICCION"* ]]; then
		echo "Copiando este fichero   $REPLY   ..." >>${LOG_VALIDADOR}
		cp $REPLY $DIR_VALIDACION
	fi
done 9< <( find $DIR_FUT_SUBGRUPOS -type f -exec printf '%s\0' {} + )

dir_val_futuro_1="/bolsa/validacion/E${NUM_EMPRESAS}_VR0_S${S}_X${X}_R${R}_M${M}_F${F}_B${B}_futuro1/"
rm -Rf $dir_val_futuro_1
mkdir -p $dir_val_futuro_1
cp -R "/bolsa/futuro/" $dir_val_futuro_1

echo -e "******** FIN  **************" >> ${LOG_VALIDADOR}




