#!/bin/bash
#set -e
echo -e "TEST INTEGRACION - INICIO: "$( date "+%Y%m%d%H%M%S" )
INFORME_OUT="/bolsa/logs/integracion.html"
echo -e  "Fichero de salida del test de integracion: ${INFORME_OUT}"
echo -e "<!DOCTYPE html><html><head><meta charset=\"UTF-8\">"  > ${INFORME_OUT}
echo -e "<style>table, th, td {  border: 1px solid black; border-collapse: collapse;}table.center {  margin-left: auto;  margin-right: auto;}</style>"  >> ${INFORME_OUT}
echo -e "</head><body>"  >> ${INFORME_OUT}
echo -e "<h2 style=\"text-align: center;\">Test de integración</h2>" >> ${INFORME_OUT}

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

############### COMPILAR JAR ########################################################
DIR_JAVA="${DIR_CODIGOS}BolsaJava/"
PATH_JAR="${DIR_JAVA}target/bolsajava-1.0-jar-with-dependencies.jar"
echo -e "Compilando JAVA en un JAR..."
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
empresa=$(cat /bolsa/pasado/subgrupos/${SG_ANALIZADO}/EMPRESAS.txt | shuf -n 1 | tr -d '\n' | cut -d'/' -f5 | cut -d'.' -f1 | cut -d'_' -f2)
echo -e "<br><b>Empresa analizada: ${empresa}</b>" >> ${INFORME_OUT}

echo -e "<br><b>FINVIZ: <a href=\"https://finviz.com/quote.ashx?t=${empresa}\">${empresa}</a></b>" >> ${INFORME_OUT}


#####
echo -e "<br><h3>Capa 1.1 (brutos desestructurados)</h3>" >> ${INFORME_OUT}
BRUTO_YF="/bolsa/pasado/brutos/YF_NASDAQ_${empresa}.txt"
BRUTO_FZ="/bolsa/pasado/brutos/FZ_NASDAQ_${empresa}.html"
echo -e "Bruto - YahooFinance: ${BRUTO_YF} --> Tamanio (bytes) = "$(stat -c%s "$BRUTO_YF") >> ${INFORME_OUT}
echo -e "<br>Bruto - Finviz - Datos estáticos: ${BRUTO_FZ} --> Tamanio (bytes) = "$(stat -c%s "$BRUTO_FZ") >> ${INFORME_OUT}

#####
echo -e "<br><h3>Capa 1.2 (brutos estructurados)</h3>" >> ${INFORME_OUT}
BRUTO_CSV="/bolsa/pasado/brutos_csv/NASDAQ_${empresa}.csv"
echo -e "Bruto CSV: <a href=\"${BRUTO_CSV}\">${BRUTO_CSV}</a> --> Tamanio (bytes) = "$(stat -c%s "$BRUTO_CSV")" con "$(wc -l $BRUTO_CSV | cut -d\  -f 1)" filas<br><br>" >> ${INFORME_OUT}
head -n 10 ${BRUTO_CSV} > "/tmp/entrada.csv"
java -jar ${PATH_JAR} --class "coordinador.Principal" "testIntegracion.ParserCsvEnTablaHtml" "/tmp/entrada.csv" "${INFORME_OUT}" "\\|" "append"

#####
echo -e "<br><h3>Capa 2 (limpios)</h3>" >> ${INFORME_OUT}
LIMPIO="/bolsa/pasado/limpios/NASDAQ_${empresa}.csv"
echo -e "Limpio: <a href=\"${LIMPIO}\">${LIMPIO}</a> --> Tamanio (bytes) = "$(stat -c%s "$LIMPIO")" con "$(wc -l $LIMPIO | cut -d\  -f 1)" filas<br><br>" >> ${INFORME_OUT}
head -n 5 ${LIMPIO}  > "/tmp/entrada.csv"
java -jar ${PATH_JAR} --class "coordinador.Principal" "testIntegracion.ParserCsvEnTablaHtml" "/tmp/entrada.csv" "${INFORME_OUT}" "\\|" "append"

#####
echo -e "<br><h3>Capa 3 (elaboradas)</h3>" >> ${INFORME_OUT}
ELABORADO="/bolsa/pasado/elaborados/NASDAQ_${empresa}.csv"
echo -e "Elaborado: <a href=\"${ELABORADO}\">${ELABORADO}</a> --> Tamanio (bytes) = "$(stat -c%s "$ELABORADO")" con "$(wc -l $ELABORADO | cut -d\  -f 1)" filas<br><br>" >> ${INFORME_OUT}
head -n 10 ${ELABORADO}  > "/tmp/entrada.csv"
java -jar ${PATH_JAR} --class "coordinador.Principal" "testIntegracion.ParserCsvEnTablaHtml" "/tmp/entrada.csv" "${INFORME_OUT}" "\\|" "append"

#####
echo -e "<br><h3>Capa 4 (subgrupos)</h3>" >> ${INFORME_OUT}
DIR_SUBGRUPOS="/bolsa/pasado/subgrupos/"
echo -e "Subgrupos creados (los que superan suficientes requisitos): <br>"$(ls $DIR_SUBGRUPOS)"<br>" >> ${INFORME_OUT}
SG_EMPRESAS="${DIR_SUBGRUPOS}${SG_ANALIZADO}/EMPRESAS.txt"

#####
echo -e "<br><h3>Capa 5 (reducir CSV)</h3><br>" >> ${INFORME_OUT}
echo -e "Flujo de datos en capa 5:     COMPLETO.csv --> Transformaciones......  --> REDUCIDO.csv<br>" >> ${INFORME_OUT}
SG_ENTRADA="${DIR_SUBGRUPOS}${SG_ANALIZADO}/COMPLETO.csv"
echo -e "Subgrupo ${SG_ANALIZADO} - Datos entrada: <a href=\"${SG_ENTRADA}\">${SG_ENTRADA}</a> --> Tamanio (bytes) = "$(stat -c%s "$SG_ENTRADA")" con "$(wc -l $SG_ENTRADA | cut -d\  -f 1)" filas de las que "$(cat ${SG_ENTRADA} | grep '^${empresa}|' | wc -l)" filas son de la empresa analizada ${empresa}.<br><br>" >> ${INFORME_OUT}

echo -e "<br>Ejemplo de filas de la empresa dentro de COMPLETO.csv:<br><br>" >> ${INFORME_OUT}
head -n 1 ${SG_ENTRADA} > "/tmp/entrada.csv"  # Solo cabecera
cat ${SG_ENTRADA} | grep "${empresa}|"  | head -n 10 >> "/tmp/entrada.csv"
java -jar ${PATH_JAR} --class "coordinador.Principal" "testIntegracion.ParserCsvEnTablaHtml" "/tmp/entrada.csv" "${INFORME_OUT}" "\\|" "append"

#####
echo -e "<br><br>Para analizar el CSV reducido (normalizado, sin nulos y solo con features dinámicas elegidas), veamos las primeras filas del fichero de entrada (COMPLETO.csv) y las primeras filas del fichero reducido (serán esas mismas filas, salvo si tenían nulos, que habrán sido eliminadas y veríamos las siguientes)<br>" >> ${INFORME_OUT}
echo -e "<br>Las primeras filas de COMPLETO (de la primera empresa que aparece):<br><br>" >> ${INFORME_OUT}
head -n 10 ${SG_ENTRADA}  > "/tmp/entrada.csv"
java -jar ${PATH_JAR} --class "coordinador.Principal" "testIntegracion.ParserCsvEnTablaHtml" "/tmp/entrada.csv" "${INFORME_OUT}" "\\|" "append"

echo -e "<br><br>Entonces, nos fijamos solo en esas features elegidas de COMPLETO:<br>" >> ${INFORME_OUT}
TMP_ENTRADA_COLUMNAS_ELEGIDAS="/tmp/temp_bolsa_testintegracion_completosoloseleccionadas.csv"
java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "testIntegracion.ExtractorFeatures" "${FEATURES_ELEGIDAS}" "${SG_ENTRADA}" "${TMP_ENTRADA_COLUMNAS_ELEGIDAS}" "10"  1>>/dev/null  2>>${INFORME_OUT}
head -n 10 "${TMP_ENTRADA_COLUMNAS_ELEGIDAS}" >> ${INFORME_OUT}

echo -e "<br>Capa 5 - Intermedio NORMALIZADO:<br>" >> ${INFORME_OUT}
echo "<b>[PENDIENTE]</b>" >> ${INFORME_OUT}

echo -e "<br><br>Capa 5 - Intermedio FEATURE SELECTION (RFE). Columnas elegidas: <br>" >> ${INFORME_OUT}
echo "<b>">> ${INFORME_OUT}
head -n 1 "${DIR_SUBGRUPOS}${SG_ANALIZADO}/FEATURES_ELEGIDAS.csv"  | sed -e "s/|/| /g" >> ${INFORME_OUT}
echo "</b>">> ${INFORME_OUT}

echo -e "<br><br>Capa 5 - Intermedio PCA. Se crea una base de funciones ortogonales como combinaciones lineales de las features de entrada. La matriz de pesos/fórmulas: <br>" >> ${INFORME_OUT}
PCA_MATRIZ="${DIR_SUBGRUPOS}${SG_ANALIZADO}/PCA_matriz.csv"
cat ${PCA_MATRIZ}  > "/tmp/entrada.csv"
java -jar ${PATH_JAR} --class "coordinador.Principal" "testIntegracion.ParserCsvEnTablaHtml" "/tmp/entrada.csv" "${INFORME_OUT}" "\\|" "append"

SG_REDUCIDO="${DIR_SUBGRUPOS}${SG_ANALIZADO}/REDUCIDO.csv"
echo -e "<br>Subgrupo ${SG_ANALIZADO} - Datos reducidos (normalizar + seleccion de columnas): <a href=\"${SG_REDUCIDO}\">${SG_REDUCIDO}</a> --> Tamanio (bytes) = "$(stat -c%s "$SG_REDUCIDO")" con "$(wc -l $SG_REDUCIDO | cut -d\  -f 1)" filas<br>" >> ${INFORME_OUT}
echo -e "<br>Y vemos la transformacion de esas filas en REDUCIDO (fijarse en si la normalización de las columnas tiene sentido!!! )<br>" >> ${INFORME_OUT}
echo -e "<b>La primera columna es el indice del dataframe. Sirve para poder identificar a que fila del COMPLETO.csv corresponde esta fila con predicciones del REDUCIDO.csv ¿Es correcto?:</b><br><br>" >> ${INFORME_OUT}
head -n 5 ${SG_REDUCIDO}  > "/tmp/entrada.csv"
java -jar ${PATH_JAR} --class "coordinador.Principal" "testIntegracion.ParserCsvEnTablaHtml" "/tmp/entrada.csv" "${INFORME_OUT}" "\\|" "append"


#####
echo -e "<br><br><h3>Capa 6 (entrenar modelo predictivo)</h3><br>" >> ${INFORME_OUT}
echo -e "<br>Subgrupo ${SG_ANALIZADO} - Modelo ganador --> "$(ls ${DIR_SUBGRUPOS}${SG_ANALIZADO}/ | grep 'ganador')"<br>" >> ${INFORME_OUT}


#######################################################################################################
echo -e "<h2>******* COMPROBACIONES del FUTURO2 ********</h2>" >> ${INFORME_OUT}
echo -e "<br><br><br><br><br>" >> ${INFORME_OUT}

echo -e "</body></html>"  >> ${INFORME_OUT}
echo -e "TEST INTEGRACION - FIN: "$( date "+%Y%m%d%H%M%S" )
