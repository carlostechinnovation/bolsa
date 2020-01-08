#!/bin/bash

################## FUNCIONES #############################################################
crearCarpetaSiNoExiste() {
	param1=${1} 			#directorio
	echo "Creando carpeta: $param1"
	mkdir -p ${param1}
}

################################################################################################
VELAS_RETROCESO="28"

# PARAMETROS DE TARGET MEDIDOS EN VELAS
S="11"
X="56"
R="7"
M="28"
F="4"

DIR_BASE="/bolsa/"

DIR_CODIGOS_CARLOS="/home/carloslinux/Desktop/GIT_BOLSA/" #NO TOCAR
DIR_CODIGOS_LUIS="/home/t151521/bolsa/" #NO TOCAR
DIR_CODIGOS="${DIR_CODIGOS_CARLOS}"

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

echo -e "PARAMETROS --> VELAS_RETROCESO|S|X|R|M|F --> ${VELAS_RETROCESO}|${S}|${X}|${R}|${M}|${F}" >>${LOG_VALIDADOR}

rm -Rf /bolsa/pasado/ >>${LOG_VALIDADOR}
mkdir -p /bolsa/logs/ >>${LOG_VALIDADOR}
echo -e ""  >>${LOG_VALIDADOR}

echo -e "Ejecución del PASADO (para entrenar los modelos a dia de HOY con la lista normal de empresas)..." >>${LOG_VALIDADOR}
${PATH_SCRIPTS}master.sh "pasado" "0" "0" "S" "${S}" "${X}" "${R}" "${M}" "${F}" 2>>${LOG_VALIDADOR} 1>>${LOG_VALIDADOR}

dir_val_pasado="/bolsa/validacion/E600_VR${VELAS_RETROCESO}_S${S}_X${X}_R${R}_M${M}_F${F}_pasado/"
rm -Rf $dir_val_pasado
mkdir -p $dir_val_pasado
cp -R "/bolsa/pasado/" $dir_val_pasado

################################################################################################

#En esta ejecucion, nos situamos unas velas atras y PREDECIMOS el futuro. Guardaremos esa predicción y la comparamos con lo que ha pasado hoy (dato REAL)

rm -Rf /bolsa/futuro/ >>${LOG_VALIDADOR}

echo -e "Ejecución del futuro (para velas de antiguedad=${VELAS_RETROCESO}) con OTRAS empresas (lista REVERTIDA)..." >>${LOG_VALIDADOR}
${PATH_SCRIPTS}master.sh "futuro" "$VELAS_RETROCESO" "1" "S" "${S}" "${X}" "${R}" "${M}" "${F}"  2>>${LOG_VALIDADOR} 1>>${LOG_VALIDADOR}

echo -e "ATRAS $VELAS_RETROCESO velas --> Guardamos la prediccion de todos los SUBGRUPOS en la carpeta de validacion, para analizarla luego..." >>${LOG_VALIDADOR}
while IFS= read -r -d '' -u 9
do
	if [[ $REPLY == *"COMPLETO_PREDICCION"* ]]; then
		echo "Copiando este fichero   $REPLY   ..." >>${LOG_VALIDADOR}
		cp $REPLY $DIR_VALIDACION
	fi
done 9< <( find $DIR_FUT_SUBGRUPOS -type f -exec printf '%s\0' {} + )

dir_val_futuro_atras="/bolsa/validacion/E600_VR${VELAS_RETROCESO}_S${S}_X${X}_R${R}_M${M}_F${F}_futuro1/"
rm -Rf $dir_val_futuro_atras
mkdir -p $dir_val_futuro_atras
cp -R "/bolsa/futuro/" $dir_val_futuro_atras

################################################################################################

#En esta ejecucion SOLO NOS INTERESA COGER el tablon analítico de entrada, para extraer el resultado REAL y compararlo con las predicciones que hicimos unas velas atrás.
#En esta ejecucion, no miramos la predicción. --> Realmente esta PREDICCION DEL FUTURO en este instante es donde vamos a PONER EL DINERO REAL. 

rm -Rf /bolsa/futuro/ >>${LOG_VALIDADOR}

vela_futuro_predicho="$((${VELAS_RETROCESO}-${M}))"

echo -e "Ejecución del futuro (para velas de anttiguedad=${vela_futuro_predicho}) con OTRAS empresas (lista REVERTIDA)..." >>${LOG_VALIDADOR}
${PATH_SCRIPTS}master.sh "futuro" "$vela_futuro_predicho" "1" "S" "${S}" "${X}" "${R}" "${M}" "${F}" 2>>${LOG_VALIDADOR} 1>>${LOG_VALIDADOR}

echo -e "VELAS_50 --> Guardamos la prediccion de todos los SUBGRUPOS en la carpeta de validacion, para analizarla luego..." >>${LOG_VALIDADOR}
while IFS= read -r -d '' -u 9
do
	if [[ $REPLY == *"COMPLETO_PREDICCION"* ]]; then
		echo "Copiando este fichero   $REPLY   ..." >>${LOG_VALIDADOR}
		cp $REPLY $DIR_VALIDACION
	fi
done 9< <( find $DIR_FUT_SUBGRUPOS -type f -exec printf '%s\0' {} + )

dir_val_futuro_2="/bolsa/validacion/E600_VR${VELAS_RETROCESO}_${S}_${X}_${R}_${M}_${F}_futuro2/"
rm -Rf $dir_val_futuro_2
mkdir -p $dir_val_futuro_2
cp -R "/bolsa/futuro/" $dir_val_futuro_2

################################################################################################
echo -e "-------------------------------------------------" >> ${LOG_VALIDADOR}
echo -e "Validacion de rendimiento: COMPRAR el ATRAS_PREDICHO con HOY_REAL para empresas de la lista REVERTIDA..." >> ${LOG_VALIDADOR}

# Supongamos que estoy en vela A (=0, actual). El sistema predictivo trabaja con el periodo [A-X, A+M], es decir, las columnas elaboradas han trabajado en un periodo de duración X+M+1.
# Nuestro validador debe calcular:
# - Pasado (entrenamiento): entrenamos suponiendo que estamos sobre la vela A=0 (vela actual), usando la lista NORMAL de EMPRESAS.
# - Futuro 1: predicción del futuro, usando la lista INVERSA de EMPRESAS, suponiendo que estamos justo encima de la vela (A - VELAS_RETROCESO). Es decir, estamos prediciendo el comportamiento en la vela (A - VELAS_RETROCESO + M) !!!!
# - Futuro 2: predicción del futuro, usando la lista INVERSA de EMPRESAS, suponiendo que estamos justo encima de la vela A. Es decir, estamos prediciendo el comportamiento en la vela (A + M). No podemos usar ese campo "predicho" (target) !!!!  Sino que debemos ver si se cumplieron las condiciones deseadas en (A - VELAS_RETROCESO + M).

# Sólo estaría bien en un caso: si VELAS_RETROCESO == M !!!   ===> Por sencillez, voy a ejecutar así.

# Lo que estabamos haciendo era comparar los campos "target" del caso "futuro 1", con los "target" del caso "futuro 2". No tiene sentido.

# Lo que realmente queremos hacer es comparar estos dos:
# - FUTURO1 --> "Usando los datos hasta la vela (A - VELAS_RETROCESO): target con lista INVERSA de empresas, que es lo predicho para la vela (A - VELAS_RETROCESO + M)".
# - FUTURO2 --> "Usando los datos hasta la vela (A - VELAS_RETROCESO + M): cumplimiento de condiciones previstas, que miran ciertas cosas en el periodo [ A - VELAS_RETROCESO - X, A - VELAS_RETROCESO + M], habiendo usado la lista INVERSA de empresas".

echo -e "-------" >> ${LOG_VALIDADOR}
echo -e "Para poder calcular la RENTABILIDAD comparando predicho y real, los parámetros de entrada siempre deben ser X<=VELAS_RETROCESO, es decir: ${X} <= ${VELAS_RETROCESO}" >> ${LOG_VALIDADOR}
echo -e "-------" >> ${LOG_VALIDADOR}
java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c70X.validacion.Validador" "${VELAS_RETROCESO}" "${DIR_VALIDACION}" "${S}" "${X}" "${R}" "${M}" 2>>${LOG_VALIDADOR} 1>>${LOG_VALIDADOR}

echo -e "Un sistema CLASIFICADOR BINOMIAL tonto acierta el 50% de las veces. El nuestro..." >> ${LOG_VALIDADOR}

dir_val_logs="/bolsa/validacion/E600_VR${VELAS_RETROCESO}_S${S}_X${X}_R${R}_M${M}_F${F}_log/"
rm -Rf $dir_val_logs
mkdir -p $dir_val_logs
cp -R "/bolsa/logs/" $dir_val_logs

echo -e "******** FIN de validacion **************" >> ${LOG_VALIDADOR}




