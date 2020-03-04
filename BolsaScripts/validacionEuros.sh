#!/bin/bash

echo -e "VALIDACION - INICIO: "$( date "+%Y%m%d%H%M%S" )

#################### DIRECTORIOS ###############################################################
DIR_CODIGOS_CARLOS="/home/carloslinux/Desktop/GIT_BOLSA/"
DIR_CODIGOS_LUIS="/home/t151521/bolsa/"
PYTHON_MOTOR_CARLOS="/home/carloslinux/Desktop/PROGRAMAS/anaconda3/envs/BolsaPython/bin/python"
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
PARAMS_CONFIG="${PATH_SCRIPTS}parametros.config"
echo -e "Importando parametros generales..."
source ${PARAMS_CONFIG}

VELAS_RETROCESO="$((${X}+${M}+2))" #INSTANTE ANALIZADO (T1). Su antiguedad debe ser mayor o igual que X+M, para poder ver esas X+M velas del futuro

#Instantes de las descargas
PASADO_t1="0"
FUTURO1_t1="${VELAS_RETROCESO}" #No tocar. Se usa en el validador
FUTURO2_t1="$((${VELAS_RETROCESO}-${X}-${M}))" #No tocar. Se usa en el validador

DIR_BASE="/bolsa/"
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

echo -e "PARAMETROS --> VELAS_RETROCESO|S|X|R|M|F|B|NUM_EMPRESAS|SUBIDA_MAXIMA_POR_VELA|MIN_COBERTURA_CLUSTER}|MIN_EMPRESAS_POR_CLUSTER|MAX_NUM_FEAT_REDUCIDAS --> ${VELAS_RETROCESO}|${S}|${X}|${R}|${M}|${F}|${B}|${NUM_EMPRESAS}|${UMBRAL_SUBIDA_POR_VELA}|${MIN_COBERTURA_CLUSTER}|${MIN_EMPRESAS_POR_CLUSTER}|${MAX_NUM_FEAT_REDUCIDAS}" >>${LOG_VALIDADOR}
echo -e "PARAMETROS -->  PASADO_t1 | FUTURO1_t1 | FUTURO2_t1--> ${PASADO_t1}|${FUTURO1_t1}|${FUTURO2_t1}" >>${LOG_VALIDADOR}

rm -Rf /bolsa/pasado/ >>${LOG_VALIDADOR}
mkdir -p /bolsa/logs/ >>${LOG_VALIDADOR}
echo -e ""  >>${LOG_VALIDADOR}

if [ "${ACTIVAR_DESCARGAS}" = "N" ];  then
	echo -e "PASADO - Usamos datos LOCALES (sin Internet) de la ruta /bolsa/datos_validacion/" >>${LOG_VALIDADOR}
	crearCarpetaSiNoExiste "/bolsa/pasado/brutos_csv/"
	cp -a "/bolsa/validacion_datos/pasado_brutos_csv/." "/bolsa/pasado/brutos_csv/" >>${LOG_VALIDADOR}
fi;

echo -e $( date '+%Y%m%d_%H%M%S' )" Ejecución del PASADO (para entrenar los modelos a dia de HOY con la lista normal de empresas)..." >>${LOG_VALIDADOR}
${PATH_SCRIPTS}master.sh "pasado" "${PASADO_t1}" "0" "${ACTIVAR_DESCARGAS}" "S" "${S}" "${X}" "${R}" "${M}" "${F}" "${B}" "${NUM_EMPRESAS}" "${UMBRAL_SUBIDA_POR_VELA}" "${MIN_COBERTURA_CLUSTER}" "${MIN_EMPRESAS_POR_CLUSTER}" "${P_INICIO}" "${P_FIN}" "${MAX_NUM_FEAT_REDUCIDAS}" 2>>${LOG_VALIDADOR} 1>>${LOG_VALIDADOR}

dir_val_pasado="/bolsa/validacion/E${NUM_EMPRESAS}_VR${VELAS_RETROCESO}_S${S}_X${X}_R${R}_M${M}_F${F}_B${B}_pasado/"
rm -Rf $dir_val_pasado
mkdir -p $dir_val_pasado
cp -R "/bolsa/pasado/" $dir_val_pasado

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
${PATH_SCRIPTS}master.sh "futuro" "$FUTURO1_t1" "1" "${ACTIVAR_DESCARGAS}" "S" "${S}" "${X}" "${R}" "${M}" "${F}" "${B}" "${NUM_EMPRESAS}" "${UMBRAL_SUBIDA_POR_VELA}" "${MIN_COBERTURA_CLUSTER}" "${MIN_EMPRESAS_POR_CLUSTER}" "20001111" "20991111" "${MAX_NUM_FEAT_REDUCIDAS}" 2>>${LOG_VALIDADOR} 1>>${LOG_VALIDADOR}

echo -e "Atrás $VELAS_RETROCESO velas --> Guardamos la prediccion de todos los SUBGRUPOS en la carpeta de validacion, para analizarla luego..." >>${LOG_VALIDADOR}
while IFS= read -r -d '' -u 9
do
	if [[ $REPLY == *"COMPLETO_PREDICCION"* ]]; then
		echo "Copiando este fichero   $REPLY   ..." >>${LOG_VALIDADOR}
		cp $REPLY $DIR_VALIDACION
	fi
done 9< <( find $DIR_FUT_SUBGRUPOS -type f -exec printf '%s\0' {} + )

dir_val_futuro_1="/bolsa/validacion/E${NUM_EMPRESAS}_VR${VELAS_RETROCESO}_S${S}_X${X}_R${R}_M${M}_F${F}_B${B}_futuro1/"
rm -Rf $dir_val_futuro_1
mkdir -p $dir_val_futuro_1
cp -R "/bolsa/futuro/" $dir_val_futuro_1

################################################################################################

#En esta ejecucion SOLO NOS INTERESA COGER el tablon analítico de entrada, para extraer el resultado REAL y compararlo con las predicciones que hicimos unas velas atrás.
#En esta ejecucion, no miramos la predicción. --> Esta NO es la predicción del futuro en el instante de ahora mismo (t1=0) para poner dinero real, sino de del futuro de que hemos predicho en la ejecucion de arriba, es decir, t1= -1* VELAS_RETROCESO + S + M

rm -Rf /bolsa/futuro/ >>${LOG_VALIDADOR}
crearCarpetaSiNoExiste "${DIR_FUT_SUBGRUPOS}"

if [ "${ACTIVAR_DESCARGAS}" = "N" ];  then
	echo -e "FUTURO2 - Usamos datos LOCALES (sin Internet) de la ruta /bolsa/datos_validacion/" >>${LOG_VALIDADOR}
	crearCarpetaSiNoExiste "/bolsa/futuro/brutos_csv/"
	cp -a "/bolsa/validacion_datos/futuro2_brutos_csv/." "/bolsa/futuro/brutos_csv/" >>${LOG_VALIDADOR}
fi;

echo -e $( date '+%Y%m%d_%H%M%S' )" Ejecución del futuro (para velas de antiguedad=${FUTURO2_t1}) con OTRAS empresas (lista REVERTIDA)..." >>${LOG_VALIDADOR}
${PATH_SCRIPTS}master.sh "futuro" "${FUTURO2_t1}" "1" "${ACTIVAR_DESCARGAS}" "S" "${S}" "${X}" "${R}" "${M}" "${F}" "${B}" "${NUM_EMPRESAS}" "${UMBRAL_SUBIDA_POR_VELA}" "${MIN_COBERTURA_CLUSTER}" "${MIN_EMPRESAS_POR_CLUSTER}" "20001111" "20991111" "${MAX_NUM_FEAT_REDUCIDAS}" 2>>${LOG_VALIDADOR} 1>>${LOG_VALIDADOR}

echo -e "Velas con antiguedad=FUTURO2_t1 --> Guardamos la prediccion de todos los SUBGRUPOS en la carpeta de validacion, para analizarla luego..." >>${LOG_VALIDADOR}
while IFS= read -r -d '' -u 9
do
	if [[ $REPLY == *"COMPLETO_PREDICCION"* ]]; then
		echo "Copiando este fichero   $REPLY   ..." >>${LOG_VALIDADOR}
		cp $REPLY $DIR_VALIDACION
	fi
done 9< <( find $DIR_FUT_SUBGRUPOS -type f -exec printf '%s\0' {} + )

dir_val_futuro_2="/bolsa/validacion/E${NUM_EMPRESAS}_VR${VELAS_RETROCESO}_S${S}_X${X}_R${R}_M${M}_F${F}_B${B}_futuro2/"
rm -Rf $dir_val_futuro_2
mkdir -p $dir_val_futuro_2
cp -R "/bolsa/futuro/" $dir_val_futuro_2

################################################################################################
echo -e $( date '+%Y%m%d_%H%M%S' )" -------------------------------------------------" >> ${LOG_VALIDADOR}
echo -e "Validacion de rendimiento: COMPARAR el ATRAS_PREDICHO con HOY_REAL para empresas de la lista REVERTIDA..." >> ${LOG_VALIDADOR}

echo -e "Nuestro sistema se situa en el instante t1." >> ${LOG_VALIDADOR}
echo -e "Para calcular el TARGET, trabaja usando los periodos [t1,t2] (X velas hacia adelante) y [t2,t3] (M velas hacia más adelante)." >> ${LOG_VALIDADOR}
echo -e "En el script de validacion, ejecutamos:" >> ${LOG_VALIDADOR}
echo -e "- Pasado con t1=${PASADO_t1}" >> ${LOG_VALIDADOR}
echo -e "- Futuro con t1 =${FUTURO1_t1} velas atras ==> Predice el target para el instante t3 = ${FUTURO2_t1}" >> ${LOG_VALIDADOR}
echo -e "- Futuro con t1=${FUTURO2_t1} ==> Nos sirve para descargar qué pasó realmente en ese t3 que hemos predicho antes y que ahora es el t1." >> ${LOG_VALIDADOR}
echo -e "Lo CORRECTO es comparar el target PREDICHO en ejecucion futuro1 y compararlo con un target GENERADO en futuro2" >> ${LOG_VALIDADOR}

echo -e "-------" >> ${LOG_VALIDADOR}
java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c70X.validacion.Validador" "${VELAS_RETROCESO}" "${DIR_VALIDACION}" "${S}" "${X}" "${R}" "${M}" 2>>${LOG_VALIDADOR} 1>>${LOG_VALIDADOR}

echo -e "Un sistema CLASIFICADOR BINOMIAL tonto acierta el 50% de las veces. El nuestro..." >> ${LOG_VALIDADOR}

dir_val_logs="/bolsa/validacion/E${NUM_EMPRESAS}_VR${VELAS_RETROCESO}_S${S}_X${X}_R${R}_M${M}_F${F}_B${B}_umbral${UMBRAL_SUBIDA_POR_VELA}_log/"
rm -Rf $dir_val_logs
mkdir -p $dir_val_logs
cp -R "/bolsa/logs/" $dir_val_logs

echo -e "******** FIN de validacion **************" >> ${LOG_VALIDADOR}


################################################################################################
# Prediccion del FUTURO para poner dinero. Esto va a crontab
################################################################################################

#/home/carloslinux/Desktop/GIT_BOLSA/BolsaScripts/predecir_futuro_real.sh

################################################################################################



echo -e "VALIDACION - FIN: "$( date "+%Y%m%d%H%M%S" )


