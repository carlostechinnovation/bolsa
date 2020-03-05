#!/bin/bash

echo -e "TEST INTEGRACION - INICIO: "$( date "+%Y%m%d%H%M%S" )


DIR_VALIDACION="/bolsa/validacion/"
LOG_INTEGRACION="/bolsa/logs/integracion.log"
echo -e ""  > ${LOG_INTEGRACION}

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

############### COMPILAR JAR ########################################################
DIR_JAVA="${DIR_CODIGOS}BolsaJava/"
PATH_JAR="${DIR_JAVA}target/bolsajava-1.0-jar-with-dependencies.jar"
echo -e "Compilando JAVA en un JAR..." >> ${LOG_INTEGRACION}
cd "${DIR_JAVA}" >> ${LOG_INTEGRACION}
rm -Rf "${DIR_JAVA}target/" >> ${LOG_INTEGRACION}
mvn clean compile assembly:single >> ${LOG_INTEGRACION}

#######################################################################################################
DIR_DATOS_INTEGRACION="${DIR_CODIGOS}test_integracion_datos/"

echo -e "Damos por hecho que tenemos una ejecucion completa de validador (1 pasado y 2 futuros)."  >> ${LOG_INTEGRACION}
echo -e "Cogemos una empresa y vemos su evolucion en cada capa."  >> ${LOG_INTEGRACION}

echo -e "************************** Preparando escenario de INTEGRACION *******************" >> ${LOG_INTEGRACION}
echo -e "Copiando fichero de configuracion..." >> ${LOG_INTEGRACION}
rm -f "${DIR_CODIGOS}parametros.config"
cp -f "${DIR_DATOS_INTEGRACION}parametros.config" "${DIR_CODIGOS}BolsaScripts/parametros.config"
echo -e "Copiando datos de entrada (ya descargados)..." >> ${LOG_INTEGRACION}
rm -Rf "/bolsa/validacion_datos/"
crearCarpetaSiNoExisteYVaciarRecursivo "/bolsa/validacion_datos/"
cp -a "${DIR_CODIGOS}test_integracion_datos/." "/bolsa/validacion_datos/"


#######################################################################################################
echo -e "********************************Ejecucion del validador: PASADO + FUTURO1 + FUTURO2 *******************************" >> ${LOG_INTEGRACION}
echo -e "Ejecutando validador..." >> ${LOG_INTEGRACION}
${DIR_CODIGOS}BolsaScripts/validacionEuros.sh


#######################################################################################################
echo -e "******************************** COMPROBACIONES del PASADO *******************************" >> ${LOG_INTEGRACION}
empresa="ABEO"
echo -e "Empresa analizada: ${empresa}" >> ${LOG_INTEGRACION}

LOG_PASADO=$(ls /bolsa/logs/ | grep 'pasado')
echo -e "LOG_PASADO: /bolsa/logs/${LOG_PASADO}"  >> ${LOG_INTEGRACION}
cat "/bolsa/logs/${LOG_PASADO}" | grep 'PARAMETROS_MASTER' >> ${LOG_INTEGRACION}

echo -e "\n-- Capa 1 (brutos) ---" >> ${LOG_INTEGRACION}
BRUTO_YF="/bolsa/pasado/brutos/YF_NASDAQ_${empresa}.txt"
BRUTO_FZ="/bolsa/pasado/brutos/FZ_NASDAQ_${empresa}.html"
#echo -e "Bruto - YahooFinance: ${BRUTO_YF} --> Tamanio (bytes) = "$(stat -c%s "$BRUTO_YF") >> ${LOG_INTEGRACION}
#echo -e "Bruto - Finviz: ${BRUTO_FZ} --> Tamanio (bytes) = "$(stat -c%s "$BRUTO_FZ") >> ${LOG_INTEGRACION}
BRUTO_CSV="/bolsa/pasado/brutos_csv/NASDAQ_${empresa}.csv"
echo -e "Bruto CSV: ${BRUTO_CSV} --> Tamanio (bytes) = "$(stat -c%s "$BRUTO_CSV")" con "$(wc -l $BRUTO_CSV | cut -d\  -f 1)" filas" >> ${LOG_INTEGRACION}
head -n 15 ${BRUTO_CSV}>> ${LOG_INTEGRACION}

echo -e "\n-- Capa 2 (limpios) ---" >> ${LOG_INTEGRACION}
LIMPIO="/bolsa/pasado/limpios/NASDAQ_${empresa}.csv"
echo -e "Limpio: ${LIMPIO} --> Tamanio (bytes) = "$(stat -c%s "$LIMPIO")" con "$(wc -l $LIMPIO | cut -d\  -f 1)" filas\n" >> ${LOG_INTEGRACION}
head -n 5 ${LIMPIO}>> ${LOG_INTEGRACION}

echo -e "\n-- Capa 3 (elaboradas) ---" >> ${LOG_INTEGRACION}
ELABORADO="/bolsa/pasado/elaborados/NASDAQ_${empresa}.csv"
echo -e "Elaborado: ${ELABORADO} --> Tamanio (bytes) = "$(stat -c%s "$ELABORADO")" con "$(wc -l $ELABORADO | cut -d\  -f 1)" filas\n" >> ${LOG_INTEGRACION}
head -n 5 ${ELABORADO}>> ${LOG_INTEGRACION}

echo -e "\n-- Capa 4 (subgrupos) ---" >> ${LOG_INTEGRACION}
DIR_SUBGRUPOS="/bolsa/pasado/subgrupos/"
echo -e "Subgrupos creados (los que superan suficientes requisitos) -> "$(ls $DIR_SUBGRUPOS)"\n" >> ${LOG_INTEGRACION}

SG_ANALIZADO="SG_11"
echo -e "Subgrupo analizado: ${SG_ANALIZADO}" >> ${LOG_INTEGRACION}

SG_EMPRESAS="${DIR_SUBGRUPOS}${SG_ANALIZADO}/EMPRESAS.txt"
echo -e "Subgrupo ${SG_ANALIZADO} - La empresa aparece en la lista de empresas del subgrupo: --> "$(cat "$SG_EMPRESAS" | grep ${empresa}) >> ${LOG_INTEGRACION}

SG_ENTRADA="${DIR_SUBGRUPOS}${SG_ANALIZADO}/COMPLETO.csv"
echo -e "Subgrupo ${SG_ANALIZADO} - Datos entrada: ${SG_ENTRADA} --> Tamanio (bytes) = "$(stat -c%s "$SG_ENTRADA")" con "$(wc -l $SG_ENTRADA | cut -d\  -f 1)" filas\n" >> ${LOG_INTEGRACION}
echo -e "Ejemplo de filas de la empresa dentro de COMPLETO.csv:" >> ${LOG_INTEGRACION}
cat ${SG_ENTRADA} | grep "${empresa}" | head -n 4 >> ${LOG_INTEGRACION}

SG_REDUCIDO="${DIR_SUBGRUPOS}${SG_ANALIZADO}/REDUCIDO.csv"
echo -e "\nSubgrupo ${SG_ANALIZADO} - Datos reducidos (normalizar + seleccion de columnas): ${SG_REDUCIDO} --> Tamanio (bytes) = "$(stat -c%s "$SG_REDUCIDO")" con "$(wc -l $SG_REDUCIDO | cut -d\  -f 1)" filas\n" >> ${LOG_INTEGRACION}

echo -e "Para analizar el CSV reducido (normalizado, sin nulos y solo con features dinámicas elegidas), veamos las primeras filas del fichero de entrada (COMPLETO.csv) y las primeras filas del fichero reducido (serán esas mismas filas, salvo si tenían nulos, que habrán sido eliminadas y veríamos las siguientes)" >> ${LOG_INTEGRACION}
echo -e "\nLas primeras filas de COMPLETO (de la primera empresa que aparece):\n" >> ${LOG_INTEGRACION}
head -n 4 "${SG_ENTRADA}" >> ${LOG_INTEGRACION}
echo -e "\nLas features elegidas han sido:\n" >> ${LOG_INTEGRACION}
FEATURES_ELEGIDAS=$(head -n 1 ${SG_REDUCIDO})
echo -e "${FEATURES_ELEGIDAS}" >> ${LOG_INTEGRACION}

echo -e "\n\nEntonces, nos fijamos solo en esas features elegidas de COMPLETO:\n" >> ${LOG_INTEGRACION}
TMP_ENTRADA_COLUMNAS_ELEGIDAS="/tmp/temp_bolsa_testintegracion_completosoloseleccionadas.csv"
java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "testIntegracion.ExtractorFeatures" "${FEATURES_ELEGIDAS}" "${SG_ENTRADA}" "${TMP_ENTRADA_COLUMNAS_ELEGIDAS}" "10"  1>>/dev/null  2>>${LOG_INTEGRACION}
head -n 10 "${TMP_ENTRADA_COLUMNAS_ELEGIDAS}" >> ${LOG_INTEGRACION}

echo -e "\nCapa 5 - Intermedio TEMP01:\n" >> ${LOG_INTEGRACION}
head -n 10 "${SG_ENTRADA}_TEMP01" >> ${LOG_INTEGRACION}
echo -e "\nCapa 5 - Intermedio TEMP02:\n" >> ${LOG_INTEGRACION}
head -n 10 "${SG_ENTRADA}_TEMP02">> ${LOG_INTEGRACION}
echo -e "\nCapa 5 - Intermedio TEMP03:\n" >> ${LOG_INTEGRACION}
head -n 10 "${SG_ENTRADA}_TEMP03">> ${LOG_INTEGRACION}
echo -e "\nCapa 5 - Intermedio TEMP04:\n" >> ${LOG_INTEGRACION}
head -n 10 "${SG_ENTRADA}_TEMP04">> ${LOG_INTEGRACION}
echo -e "\nCapa 5 - Intermedio TEMP05:\n" >> ${LOG_INTEGRACION}
head -n 10 "${SG_ENTRADA}_TEMP05">> ${LOG_INTEGRACION}
echo -e "\nCapa 5 - Intermedio TEMP06:\n" >> ${LOG_INTEGRACION}
head -n 10 "${SG_REDUCIDO}_TEMP06">> ${LOG_INTEGRACION}

echo -e "\n\nY vemos la transformacion de esas filas en REDUCIDO (fijarse en si la normalización de las columnas tiene sentido!!! ):\n" >> ${LOG_INTEGRACION}
head -n 10 ${SG_REDUCIDO}>> ${LOG_INTEGRACION}

echo -e "\nSubgrupo ${SG_ANALIZADO} - Modelo ganador --> "$(ls ${DIR_SUBGRUPOS}${SG_ANALIZADO}/ | grep 'ganador')"\n" >> ${LOG_INTEGRACION}


#######################################################################################################
echo -e "******************************** COMPROBACIONES del FUTURO1 *******************************" >> ${LOG_INTEGRACION}



#######################################################################################################
echo -e "******************************** COMPROBACIONES del FUTURO2 *******************************" >> ${LOG_INTEGRACION}




echo -e "TEST INTEGRACION - INICIO: "$( date "+%Y%m%d%H%M%S" )
