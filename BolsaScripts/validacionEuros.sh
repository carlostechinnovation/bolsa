#!/bin/bash

################## FUNCIONES #############################################################
crearCarpetaSiNoExiste() {
	param1=${1} 			#directorio
	echo "Creando carpeta: $param1"
	mkdir -p ${param1}
}

################################################################################################
VELAS_RETROCESO="28" #INSTANTE ANALIZADO (T2). Su antiguedad siempre es mayor que M, para poder ver esas M velas del futuro

# PARAMETROS DE TARGET MEDIDOS EN VELAS
S="10"  #Subida durante [t1,t2]
X="56"  #Duracion en velas de [t1,t2]
R="8"  #Caida ligera máxima permitida durante [t2,t3], en TODAS esas velas.
M="7"  #Duración en velas de [t2,t3]
F="4"  #Caida ligera permitida durante [t2,t3], en la ÚLTIMA vela

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

echo -e "Nuestro sistema se situa en el instante t2." >> ${LOG_VALIDADOR}
echo -e "Para calcular el TARGET, trabaja usando los periodos [t1,t2] (X velas hacia atrás) y [t2,t3] (M velas hacia adelante)." >> ${LOG_VALIDADOR}
echo -e "En el script de validacion, ejecutamos:" >> ${LOG_VALIDADOR}
echo -e "- Pasado con t2=0" >> ${LOG_VALIDADOR}
echo -e "- Futuro con t2 -> 50 velas atrás ==> Predice el target para el instante t3 (= t2 - 50 +M)" >> ${LOG_VALIDADOR}
echo -e "- Futuro con t2=0  ==> Descarga los datos reales en t2=0 y predice el target para t3 ( t2 + M)" >> ${LOG_VALIDADOR}
echo -e "Lo CORRECTO es comparar el target PREDICHO para (t2 -50 +M) y compararlo con un target GENERADO mirando si se han cumplido las condiciones en (t2 -50 +M)" >> ${LOG_VALIDADOR}
echo -e "Además, es importante situar t2 en un instante del pasado (no vale justo ahora), porque tenemos que ver las M velas siguientes." >> ${LOG_VALIDADOR}
echo -e "Al menos, t2 debe estar -50+M velas más atrás que ahora mismo" >> ${LOG_VALIDADOR}

echo -e "-------" >> ${LOG_VALIDADOR}
echo -e "Para poder calcular la RENTABILIDAD comparando predicho y real, los parámetros de entrada siempre deben ser X<=VELAS_RETROCESO, es decir: ${X} <= ${VELAS_RETROCESO}" >> ${LOG_VALIDADOR}
echo -e "-------" >> ${LOG_VALIDADOR}
java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c70X.validacion.Validador" "${VELAS_RETROCESO}" "${DIR_VALIDACION}" "${S}" "${X}" "${R}" "${M}" "${F}" 2>>${LOG_VALIDADOR} 1>>${LOG_VALIDADOR}

echo -e "Un sistema CLASIFICADOR BINOMIAL tonto acierta el 50% de las veces. El nuestro..." >> ${LOG_VALIDADOR}

dir_val_logs="/bolsa/validacion/E600_VR${VELAS_RETROCESO}_S${S}_X${X}_R${R}_M${M}_F${F}_log/"
rm -Rf $dir_val_logs
mkdir -p $dir_val_logs
cp -R "/bolsa/logs/" $dir_val_logs

echo -e "******** FIN de validacion **************" >> ${LOG_VALIDADOR}




