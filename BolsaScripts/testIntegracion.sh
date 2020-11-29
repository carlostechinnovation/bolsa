#!/bin/bash
#set -e
echo -e "TEST INTEGRACION - INICIO: "$( date "+%Y%m%d%H%M%S" )
INFORME_OUT="/bolsa/logs/integracion.html"
echo -e  "Fichero de salida del test de integracion: ${INFORME_OUT}"
echo -e "<!DOCTYPE html><html><head><meta charset=\"UTF-8\"></head><body>"  > ${INFORME_OUT}

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
echo -e "Compilando JAVA en un JAR...<br>" >> ${INFORME_OUT}
cd "${DIR_JAVA}" >> ${INFORME_OUT}
rm -Rf "${DIR_JAVA}target/" >> ${INFORME_OUT}
mvn clean compile assembly:single 1>/dev/null  2>> ${INFORME_OUT}

#######################################################################################################
echo -e "Damos por hecho que tenemos una ejecucion completa de PASADO+FUT1-FUT2<br>"  >> ${INFORME_OUT}
echo -e "Cogemos una empresa y vemos su evolucion en cada capa.<br>"  >> ${INFORME_OUT}

#######################################################################################################
echo -e "<h2>******* COMPROBACIONES del PASADO ********</h2>" >> ${INFORME_OUT}
echo -e "Elegimos un subgrupo y empresa al azar para el que tengamos datos hasta la última capa...<br>" >> ${INFORME_OUT}
SG_ANALIZADO=$(find "/bolsa/pasado/subgrupos/" | grep "REDUCIDO" | shuf -n 1 | cut -d'/' -f5)
echo -e "<b>Subgrupo analizado: ${SG_ANALIZADO}</b>" >> ${INFORME_OUT}
empresa=$(cat /bolsa/pasado/subgrupos/${SG_ANALIZADO}/EMPRESAS.txt | head -n 1 | tr -d '\n' | cut -d'/' -f5 | cut -d'.' -f1 | cut -d'_' -f2)
echo -e "<br><b>Empresa analizada: ${empresa}</b>" >> ${INFORME_OUT}
#####
echo -e "<br><h3>Capa 1.1 (brutos desestructurados)</h3>" >> ${INFORME_OUT}
BRUTO_YF="/bolsa/pasado/brutos/YF_NASDAQ_${empresa}.txt"
BRUTO_FZ="/bolsa/pasado/brutos/FZ_NASDAQ_${empresa}.html"
echo -e "Bruto - YahooFinance: ${BRUTO_YF} --> Tamanio (bytes) = "$(stat -c%s "$BRUTO_YF") >> ${INFORME_OUT}
echo -e "<br>Bruto - Finviz - Datos estáticos: ${BRUTO_FZ} --> Tamanio (bytes) = "$(stat -c%s "$BRUTO_FZ") >> ${INFORME_OUT}
#####
echo -e "<br><h3>Capa 1.2 (brutos estructurados)</h3>" >> ${INFORME_OUT}
BRUTO_CSV="/bolsa/pasado/brutos_csv/NASDAQ_${empresa}.csv"
echo -e "Bruto CSV: ${BRUTO_CSV} --> Tamanio (bytes) = "$(stat -c%s "$BRUTO_CSV")" con "$(wc -l $BRUTO_CSV | cut -d\  -f 1)" filas<br><br>" >> ${INFORME_OUT}
head -n 10 ${BRUTO_CSV}  | sed -z 's/|/| /g'  | sed -z 's/\n/<br>\n/g' >> ${INFORME_OUT}
#####
echo -e "<br><h3>Capa 2 (limpios)</h3>" >> ${INFORME_OUT}
LIMPIO="/bolsa/pasado/limpios/NASDAQ_${empresa}.csv"
echo -e "Limpio: ${LIMPIO} --> Tamanio (bytes) = "$(stat -c%s "$LIMPIO")" con "$(wc -l $LIMPIO | cut -d\  -f 1)" filas<br><br>" >> ${INFORME_OUT}
head -n 5 ${LIMPIO}  | sed -z 's/|/| /g'  | sed -z 's/\n/<br>\n/g' >> ${INFORME_OUT}
#####
echo -e "<br><h3>Capa 3 (elaboradas)</h3>" >> ${INFORME_OUT}
ELABORADO="/bolsa/pasado/elaborados/NASDAQ_${empresa}.csv"
echo -e "Elaborado: ${ELABORADO} --> Tamanio (bytes) = "$(stat -c%s "$ELABORADO")" con "$(wc -l $ELABORADO | cut -d\  -f 1)" filas<br><br>" >> ${INFORME_OUT}
head -n 5 ${ELABORADO}  | sed -z 's/|/| /g'  | sed -z 's/\n/<br><br>\n/g' >> ${INFORME_OUT}
#####
echo -e "<br><h3>Capa 4 (subgrupos)</h3>" >> ${INFORME_OUT}
DIR_SUBGRUPOS="/bolsa/pasado/subgrupos/"
echo -e "Subgrupos creados (los que superan suficientes requisitos) -> "$(ls $DIR_SUBGRUPOS)"<br>" >> ${INFORME_OUT}
SG_EMPRESAS="${DIR_SUBGRUPOS}${SG_ANALIZADO}/EMPRESAS.txt"
echo -e "<br>El subgrupo analizado (elegido al azar) es: ${SG_ANALIZADO} --> La empresa analizada aparece en la lista de empresas de ese subgrupo: --> "$(cat "$SG_EMPRESAS" | grep '_${empresa}\.') >> ${INFORME_OUT}
#####
echo -e "<br><h3>Capa 5 (reducir CSV y entrenar modelo)</h3><br>" >> ${INFORME_OUT}
echo -e "Flujo de datos en capa 5:     COMPLETO.csv --> Transformaciones......  --> REDUCIDO.csv<br>" >> ${INFORME_OUT}
SG_ENTRADA="${DIR_SUBGRUPOS}${SG_ANALIZADO}/COMPLETO.csv"
echo -e "Subgrupo ${SG_ANALIZADO} - Datos entrada: ${SG_ENTRADA} --> Tamanio (bytes) = "$(stat -c%s "$SG_ENTRADA")" con "$(wc -l $SG_ENTRADA | cut -d\  -f 1)" filas de las que "$(cat ${SG_ENTRADA} | grep '^${empresa}|' | wc -l)" filas son de la empresa analizada ${empresa}.<br><br>" >> ${INFORME_OUT}




echo -e "<br>Ejemplo de filas de la empresa dentro de COMPLETO.csv:<br><br>" >> ${INFORME_OUT}
head -n 2 ${SG_ENTRADA}   | sed -z 's/|/| /g'  | sed -z 's/\n/<br><br>\n/g'   >> ${INFORME_OUT}  # Cabecera
cat ${SG_ENTRADA} | grep "${empresa}" | head -n 4   | sed -z 's/|/| /g' | sed -z 's/\n/<br><br>\n/g'  >> ${INFORME_OUT} #Datos
#####
echo -e "<br><br>Para analizar el CSV reducido (normalizado, sin nulos y solo con features dinámicas elegidas), veamos las primeras filas del fichero de entrada (COMPLETO.csv) y las primeras filas del fichero reducido (serán esas mismas filas, salvo si tenían nulos, que habrán sido eliminadas y veríamos las siguientes)<br>" >> ${INFORME_OUT}
echo -e "<br>Las primeras filas de COMPLETO (de la primera empresa que aparece):<br><br>" >> ${INFORME_OUT}
head -n 4 "${SG_ENTRADA}"  | sed -z 's/|/| /g'  | sed -z 's/\n/<br><br>\n/g' >> ${INFORME_OUT}

SG_REDUCIDO="${DIR_SUBGRUPOS}${SG_ANALIZADO}/REDUCIDO.csv"
echo -e "<br><br>Las features elegidas han sido:<br>" >> ${INFORME_OUT}
FEATURES_ELEGIDAS=$(head -n 1 ${SG_REDUCIDO}  | sed -z 's/|/| /g' )
echo -e "${FEATURES_ELEGIDAS}" >> ${INFORME_OUT}

echo -e "<br><br>Entonces, nos fijamos solo en esas features elegidas de COMPLETO:<br>" >> ${INFORME_OUT}
TMP_ENTRADA_COLUMNAS_ELEGIDAS="/tmp/temp_bolsa_testintegracion_completosoloseleccionadas.csv"
java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "testIntegracion.ExtractorFeatures" "${FEATURES_ELEGIDAS}" "${SG_ENTRADA}" "${TMP_ENTRADA_COLUMNAS_ELEGIDAS}" "10"  1>>/dev/null  2>>${INFORME_OUT}
head -n 10 "${TMP_ENTRADA_COLUMNAS_ELEGIDAS}" >> ${INFORME_OUT}

echo -e "<br>Capa 5 - Intermedio TEMP01:<br>" >> ${INFORME_OUT}
head -n 10 "${SG_ENTRADA}_TEMP01" >> ${INFORME_OUT}
echo -e "<br>Capa 5 - Intermedio TEMP02:<br>" >> ${INFORME_OUT}
head -n 10 "${SG_ENTRADA}_TEMP02">> ${INFORME_OUT}
echo -e "<br>Capa 5 - Intermedio TEMP03:<br>" >> ${INFORME_OUT}
head -n 10 "${SG_ENTRADA}_TEMP03">> ${INFORME_OUT}
echo -e "<br>Capa 5 - Intermedio TEMP04:<br>" >> ${INFORME_OUT}
head -n 10 "${SG_ENTRADA}_TEMP04">> ${INFORME_OUT}
echo -e "<br>Capa 5 - Intermedio TEMP05:<br>" >> ${INFORME_OUT}
head -n 10 "${SG_ENTRADA}_TEMP05">> ${INFORME_OUT}
echo -e "<br>Capa 5 - Intermedio TEMP06:<br>" >> ${INFORME_OUT}
head -n 10 "${SG_REDUCIDO}_TEMP06">> ${INFORME_OUT}


echo -e "<br>Subgrupo ${SG_ANALIZADO} - Datos reducidos (normalizar + seleccion de columnas): ${SG_REDUCIDO} --> Tamanio (bytes) = "$(stat -c%s "$SG_REDUCIDO")" con "$(wc -l $SG_REDUCIDO | cut -d\  -f 1)" filas<br>" >> ${INFORME_OUT}

echo -e "<br><br>Y vemos la transformacion de esas filas en REDUCIDO (fijarse en si la normalización de las columnas tiene sentido!!! ) <br><b>La primera columna es el indice del dataframe. Sirve para poder identificar a que fila del COMPLETO.csv corresponde esta fila con predicciones del REDUCIDO.csv ¿Es correcto?:</b><br><br>" >> ${INFORME_OUT}
head -n 5 ${SG_REDUCIDO}  | sed -z 's/|/| /g'  | sed -z 's/\n/<br><br>\n/g' >> ${INFORME_OUT}

echo -e "<br>Subgrupo ${SG_ANALIZADO} - Modelo ganador --> "$(ls ${DIR_SUBGRUPOS}${SG_ANALIZADO}/ | grep 'ganador')"<br>" >> ${INFORME_OUT}


#######################################################################################################
echo -e "<h2>******* COMPROBACIONES del FUTURO2 ********</h2>" >> ${INFORME_OUT}


echo -e "</body></html>"  >> ${INFORME_OUT}
echo -e "TEST INTEGRACION - FIN: "$( date "+%Y%m%d%H%M%S" )
