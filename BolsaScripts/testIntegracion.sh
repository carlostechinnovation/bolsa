#!/bin/bash
#set -e
echo -e "TEST INTEGRACION - INICIO: "$( date "+%Y%m%d%H%M%S" )

INFORME_OUT="/bolsa/logs/integracion.html"
echo -e  "Fichero de salida del test de integracion: ${INFORME_OUT}"
echo -e "<!DOCTYPE html><html><head><meta charset=\"UTF-8\">"  > ${INFORME_OUT}
echo -e "<style>table, th, td {  border: 1px solid black; border-collapse: collapse;}table.center {  margin-left: auto;  margin-right: auto;}</style>"  >> ${INFORME_OUT}
echo -e "</head><body>"  >> ${INFORME_OUT}
echo -e "<h2 style=\"text-align: center;\">Test de integración</h2>" >> ${INFORME_OUT}

############# parametros de entrada ########
if [ $# -eq 0 ];  then
    echo "Hay 0 parametros de entrada. Se elegiran subgrupo+empresa al azar."
	#Elegimos un subgrupo y empresa al azar para el que tengamos datos hasta la última capa...
	SG_ANALIZADO=$(find "/bolsa/pasado/subgrupos/" | grep "REDUCIDO" | shuf -n 1 | cut -d'/' -f5)
	empresa=$(cat /bolsa/pasado/subgrupos/SG_${SG_ANALIZADO}/EMPRESAS.txt | shuf -n 1 | tr -d '\n' | cut -d'/' -f5 | cut -d'.' -f1 | cut -d'_' -f2)
	
elif [ $# -eq 2 ];  then
	echo "Hay 2 parametros de entrada: SUBGRUPO + EMPRESA del pasado."
	SG_ANALIZADO="${1}"
	empresa="${2}"
else
    echo "INSTRUCCIONES:   RUTA/script.sh  subgrupo empresa"
	echo "EJEMPLO:    /home/carloslinux/Desktop/GIT_BOLSA/BolsaScripts/testIntegracion.sh  SG_49 AAPL"
	echo "El número de parametros de entrada no es el esperado. Saliendo..."
	exit -1
fi

echo "PASADO - Subgrupo: ${SG_ANALIZADO}"
echo "PASADO - Empresa: ${empresa}"

#################### DIRECTORIOS ###############################################################
DIR_CODIGOS_CARLOS="/home/carloslinux/Desktop/GIT_BOLSA/"
DIR_CODIGOS_LUIS="/home/t151521/bolsa/"
PYTHON_MOTOR_CARLOS="/home/carloslinux/anaconda3/envs/BolsaPython38/bin/python"
PYTHON_MOTOR_LUIS="/usr/bin/python3.8"

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

PYTHON_SCRIPTS="${DIR_CODIGOS}BolsaPython/"

################## FUNCIONES #############################################################

function buscarEmpresaEnSubgrupos () {
	dirSubgrupos="${1}"
	empresa="${2}"
	ficheroLog="${3}"
	cadena=""
	for nombreSubg in $(ls ${dirSubgrupos})
	do
		pathEmpresasDeSg="/bolsa/pasado/subgrupos/${nombreSubg}/EMPRESAS.txt"
		
		if [ -f "$pathEmpresasDeSg" ]; then
			echo "Analizando si la empresa ${empresa} esta en el fichero ${pathEmpresasDeSg} ..."
			pattern="_${empresa}\."
			encontrado=$(cat ${pathEmpresasDeSg} | grep "$pattern")
			if [ -z "${encontrado}" ];then
				echo "$encontrado is empty"
			else
				echo "$encontrado is NOT empty"
				cadena="${cadena} ${nombreSubg}"
			fi
		fi
		
		
	done
	
	echo "La empresa aparece en estos subgrupos: ${cadena}" >> "${ficheroLog}"
}

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
echo -e "<b>Subgrupo analizado: ${SG_ANALIZADO}</b>" >> ${INFORME_OUT}
echo -e "<br><b>Empresa analizada: ${empresa}</b>" >> ${INFORME_OUT}
echo -e "<br><b>FINVIZ: <a href=\"https://finviz.com/quote.ashx?t=${empresa}\">${empresa}</a></b>" >> ${INFORME_OUT}

#####
echo -e "<br><h3>Capa 1.1 (brutos desestructurados)</h3>" >> ${INFORME_OUT}
BRUTO_YF="/bolsa/pasado/brutos/YF_NASDAQ_${empresa}.txt"
BRUTO_FZ="/bolsa/pasado/brutos/FZ_NASDAQ_${empresa}.html"
echo -e "Bruto - YahooFinance: <a href=\"${BRUTO_YF}\">${BRUTO_YF}</a> --> Tamanio (bytes) = "$(stat -c%s "$BRUTO_YF") >> ${INFORME_OUT}
echo -e "<br>Bruto - Finviz - Datos estáticos: <a href=\"${BRUTO_FZ}\">${BRUTO_FZ}</a> --> Tamanio (bytes) = "$(stat -c%s "$BRUTO_FZ") >> ${INFORME_OUT}

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
echo -e "Viendo la columna TARGET, este es el numero de casos de cada tipo: " >> ${INFORME_OUT}
echo -e "--> NULL = "$(cat /bolsa/pasado/elaborados/NASDAQ_AMRS.csv | grep '|null$' | wc -l) >> ${INFORME_OUT}
echo -e "  True ="$(cat /bolsa/pasado/elaborados/NASDAQ_AMRS.csv | grep '|1$' | wc -l) >> ${INFORME_OUT}
echo -e "  False = "$(cat /bolsa/pasado/elaborados/NASDAQ_AMRS.csv | grep '|0$' | wc -l) >> ${INFORME_OUT}
head -n 10 ${ELABORADO}  > "/tmp/entrada.csv"
java -jar ${PATH_JAR} --class "coordinador.Principal" "testIntegracion.ParserCsvEnTablaHtml" "/tmp/entrada.csv" "${INFORME_OUT}" "\\|" "append"

#####
echo -e "<br><h3>Capa 4 (subgrupos)</h3>" >> ${INFORME_OUT}
DIR_SUBGRUPOS="/bolsa/pasado/subgrupos/"
buscarEmpresaEnSubgrupos "${DIR_SUBGRUPOS}" "${empresa}" "${INFORME_OUT}"


#####
echo -e "<br><h3>Capa 5 (reducir CSV)</h3><br>" >> ${INFORME_OUT}
SG_ENTRADA="${DIR_SUBGRUPOS}SG_${SG_ANALIZADO}/intermedio.csv.entrada"
num_filas_empresa_en_completo=$(grep ${empresa} ${SG_ENTRADA} | wc -l)
echo -e "Fichero intermedio: <a href=\"${SG_ENTRADA}\">${SG_ENTRADA}</a> --> Tamanio (bytes) = "$(stat -c%s "$SG_ENTRADA")" con "$(wc -l $SG_ENTRADA | cut -d\  -f 1)" filas de las que "${num_filas_empresa_en_completo}" filas son de la empresa analizada "${empresa}".<br>" >> ${INFORME_OUT}

echo -e "<br>Contiene (filas de ejemplo):<br><br>" >> ${INFORME_OUT}
head -n 1 ${SG_ENTRADA} > "/tmp/entrada.csv"  # Solo cabecera
cat ${SG_ENTRADA} | grep "${empresa}|"  | head -n 10 >> "/tmp/entrada.csv"
java -jar ${PATH_JAR} --class "coordinador.Principal" "testIntegracion.ParserCsvEnTablaHtml" "/tmp/entrada.csv" "${INFORME_OUT}" "\\|" "append"


#####
SG_ENTRADA_UMBRAL="${DIR_SUBGRUPOS}SG_${SG_ANALIZADO}/intermedio.csv.entrada_tras_maximo.csv"
SG_ENTRADA_UMBRAL_INDICES="${DIR_SUBGRUPOS}SG_${SG_ANALIZADO}/intermedio.csv.entrada_tras_maximo_INDICES.csv"
num_filas_empresa_en_completo=$(grep ${empresa} ${SG_ENTRADA_UMBRAL} | wc -l)
echo -e "<br>Fichero tras umbral maximo de filas: <a href=\"${SG_ENTRADA_UMBRAL}\">${SG_ENTRADA_UMBRAL}</a> --> Tamanio (bytes) = "$(stat -c%s "$SG_ENTRADA_UMBRAL")" con "$(wc -l $SG_ENTRADA_UMBRAL | cut -d\  -f 1)" filas de las que "${num_filas_empresa_en_completo}" filas son de la empresa analizada "${empresa}".<br>" >> ${INFORME_OUT}

echo -e "<br>Ejemplos <b>(tras umbral maximo de filas)</b>:<br><br>" >> ${INFORME_OUT}
head -n 1 ${SG_ENTRADA_UMBRAL} > "/tmp/entrada.csv"  # Solo cabecera
cat ${SG_ENTRADA_UMBRAL} | grep "${empresa}|"  | head -n 10 >> "/tmp/entrada.csv"
java -jar ${PATH_JAR} --class "coordinador.Principal" "testIntegracion.ParserCsvEnTablaHtml" "/tmp/entrada.csv" "${INFORME_OUT}" "\\|" "append"


#####
SG_ENTRADA_SOLOCOMPLETAS="${DIR_SUBGRUPOS}SG_${SG_ANALIZADO}/intermedio.csv.sololascompletas.csv"
SG_ENTRADA_SC_INDICES="${DIR_SUBGRUPOS}SG_${SG_ANALIZADO}/intermedio.csv.sololascompletas_INDICES.csv"
echo -e "<br>Fichero (solo filas completas, sin ningun NaN ni nulo y SIN IDENTIFICADORES) (se han quitado las filas con target nulo si las hubiera): <a href=\"${SG_ENTRADA_SOLOCOMPLETAS}\">${SG_ENTRADA_SOLOCOMPLETAS}</a> --> Tamanio (bytes) = "$(stat -c%s "$SG_ENTRADA_SOLOCOMPLETAS")" con "$(wc -l $SG_ENTRADA_SOLOCOMPLETAS | cut -d\  -f 1)" filas.<br>" >> ${INFORME_OUT}

echo -e "<br>Ejemplos de CUALQUIER empresa del subgrupo<b> (solo filas completas, sin ningun NaN ni nulo y SIN IDENTIFICADORES) (se han quitado las filas con target nulo si las hubiera)</b>:<br><br>" >> ${INFORME_OUT}
head -n 10 ${SG_ENTRADA_SOLOCOMPLETAS} > "/tmp/entrada.csv"
java -jar ${PATH_JAR} --class "coordinador.Principal" "testIntegracion.ParserCsvEnTablaHtml" "/tmp/entrada.csv" "${INFORME_OUT}" "\\|" "append"


#####
SG_ENTRADA_SINOUTLIERS="${DIR_SUBGRUPOS}SG_${SG_ANALIZADO}/intermedio.csv.sinoutliers.csv"
SG_ENTRADA_SINOUTL_INDICES="${DIR_SUBGRUPOS}SG_${SG_ANALIZADO}/intermedio.csv.sinoutliers_INDICES.csv"
echo -e "<br>Fichero (sin outliers y SIN IDENTIFICADORES): <a href=\"${SG_ENTRADA_SINOUTLIERS}\">${SG_ENTRADA_SINOUTLIERS}</a> --> Tamanio (bytes) = "$(stat -c%s "$SG_ENTRADA_SINOUTLIERS")" con "$(wc -l $SG_ENTRADA_SINOUTLIERS | cut -d\  -f 1)" filas<br>" >> ${INFORME_OUT}

echo -e "<br>Ejemplos de CUALQUIER empresa del subgrupo<b> (sin outliers y SIN IDENTIFICADORES)</b>:<br><br>" >> ${INFORME_OUT}
echo "" > "/tmp/entrada.csv"
head -n 10 ${SG_ENTRADA_SINOUTLIERS}   > "/tmp/entrada.csv"
java -jar ${PATH_JAR} --class "coordinador.Principal" "testIntegracion.ParserCsvEnTablaHtml" "/tmp/entrada.csv" "${INFORME_OUT}" "\\|" "append"



#####
echo -e "<br><b>NORMALIZADO por columnas ( ¡¡¡¡¡¡ Y se han quitado las filas con al menos un campo NULO !!!!  )</b>:<br>" >> ${INFORME_OUT}
SG_NORMALIZADO="${DIR_SUBGRUPOS}SG_${SG_ANALIZADO}/intermedio.csv.normalizado.csv"
echo -e "Limpio: <a href=\"${SG_NORMALIZADO}\">${SG_NORMALIZADO}</a> --> Tamanio (bytes) = "$(stat -c%s "$SG_NORMALIZADO")" con "$(wc -l $SG_NORMALIZADO | cut -d\  -f 1)" filas<br><br>" >> ${INFORME_OUT}
echo -e "Hacemos el cruce con Python para coger solo los indices de la empresa analizada..." >> ${INFORME_OUT}
SG_NORMALIZADO_CRUZADO="${SG_NORMALIZADO}.cruzado.csv"
$PYTHON_MOTOR "${PYTHON_SCRIPTS}bolsa/TestIntegracionUtilidad1.py" "${SG_NORMALIZADO}" "${SG_ENTRADA_UMBRAL}" "${empresa}" "${SG_NORMALIZADO_CRUZADO}"
head -n 10 ${SG_NORMALIZADO_CRUZADO}  > "/tmp/entrada.csv"
java -jar ${PATH_JAR} --class "coordinador.Principal" "testIntegracion.ParserCsvEnTablaHtml" "/tmp/entrada.csv" "${INFORME_OUT}" "\\|" "append"



echo -e "<br><br>FEATURES UTILIZADAS: <br>" >> ${INFORME_OUT}
echo "<b>">> ${INFORME_OUT}
head -n 1 "${DIR_SUBGRUPOS}SG_${SG_ANALIZADO}/FEATURES_ELEGIDAS_RFECV.csv"  | sed -e "s/|/| /g" >> ${INFORME_OUT}
echo "</b>">> ${INFORME_OUT}

echo -e "<br><br>PCA ==> Se crea una base de funciones ortogonales como combinaciones lineales de las features de entrada. La matriz de pesos/fórmulas: <br><br>" >> ${INFORME_OUT}
PCA_MATRIZ="${DIR_SUBGRUPOS}SG_${SG_ANALIZADO}/PCA_matriz.csv"
cat ${PCA_MATRIZ}  > "/tmp/entrada.csv"
java -jar ${PATH_JAR} --class "coordinador.Principal" "testIntegracion.ParserCsvEnTablaHtml" "/tmp/entrada.csv" "${INFORME_OUT}" "\\|" "append"


SG_REDUCIDO="${DIR_SUBGRUPOS}SG_${SG_ANALIZADO}/REDUCIDO.csv"
echo -e "<br>Datos reducidos (normalizar + seleccion de columnas): <a href=\"${SG_REDUCIDO}\">${SG_REDUCIDO}</a> --> Tamanio (bytes) = "$(stat -c%s "$SG_REDUCIDO")" con "$(wc -l $SG_REDUCIDO | cut -d\  -f 1)" filas<br>" >> ${INFORME_OUT}
echo -e "<br>Y vemos la transformacion de esas filas en REDUCIDO (fijarse en si la normalización de las columnas tiene sentido!!! )<br>" >> ${INFORME_OUT}
echo -e "Hacemos el cruce con Python para coger solo los indices de la empresa analizada..." >> ${INFORME_OUT}
SG_REDUCIDO_CRUZADO="${SG_REDUCIDO}.cruzado.csv"
$PYTHON_MOTOR "${PYTHON_SCRIPTS}bolsa/TestIntegracionUtilidad1.py" "${SG_REDUCIDO}" "${SG_ENTRADA_UMBRAL}" "${empresa}" "${SG_REDUCIDO_CRUZADO}"
echo -e "<b>La primera columna es el indice del dataframe. Sirve para poder identificar a que fila del COMPLETO.csv corresponde esta fila con predicciones del REDUCIDO.csv ¿Es correcto?:</b><br><br>" >> ${INFORME_OUT}
head -n 10 ${SG_REDUCIDO_CRUZADO}  > "/tmp/entrada.csv"
java -jar ${PATH_JAR} --class "coordinador.Principal" "testIntegracion.ParserCsvEnTablaHtml" "/tmp/entrada.csv" "${INFORME_OUT}" "\\|" "append"


#####
SG_ENTRADA_CONBALANCEO="${DIR_SUBGRUPOS}SG_${SG_ANALIZADO}/intermedio.csv.trasbalancearclases.csv"
SG_ENTRADA_CONBALANCEO_INDICES="${DIR_SUBGRUPOS}SG_${SG_ANALIZADO}/intermedio.csv.trasbalancearclases_INDICES.csv"
echo -e "<br>Datos con BALANCEO de clases: <a href=\"${SG_ENTRADA_CONBALANCEO}\">${SG_ENTRADA_CONBALANCEO}</a> --> Tamanio (bytes) = "$(stat -c%s "$SG_ENTRADA_CONBALANCEO")" con "$(wc -l $SG_ENTRADA_CONBALANCEO | cut -d\  -f 1)" filas<br>" >> ${INFORME_OUT}

echo -e "<br>Ejemplos de CUALQUIER empresa del subgrupo <b>(con BALANCEO de clases)</b>:<br><br>" >> ${INFORME_OUT}
head -n 10 ${SG_ENTRADA_CONBALANCEO}  > "/tmp/entrada.csv"
java -jar ${PATH_JAR} --class "coordinador.Principal" "testIntegracion.ParserCsvEnTablaHtml" "/tmp/entrada.csv" "${INFORME_OUT}" "\\|" "append"



#####
SG_PRECISION_TRAIN_TARGETS_REALES="${DIR_SUBGRUPOS}SG_${SG_ANALIZADO}/intermedio.csv.ds_train_t_sinsmote.csv"
echo -e "<br>Dataset TRAIN - Targets reales (SIN SMOTE): <a href=\"${SG_PRECISION_TRAIN_TARGETS_REALES}\">${SG_PRECISION_TRAIN_TARGETS_REALES}</a> --> "$(wc -l $SG_PRECISION_TRAIN_TARGETS_REALES | cut -d\  -f 1)" filas ("$(cat $SG_PRECISION_TRAIN_TARGETS_REALES  | grep 'True' |wc -l)" positivos)<br>" >> ${INFORME_OUT}

echo -e "<br>Ejemplos <b>(train - targets reales SIN SMOTE)</b>:<br><br>" >> ${INFORME_OUT}
head -n 10 ${SG_PRECISION_TRAIN_TARGETS_REALES}  > "/tmp/entrada.csv"
java -jar ${PATH_JAR} --class "coordinador.Principal" "testIntegracion.ParserCsvEnTablaHtml" "/tmp/entrada.csv" "${INFORME_OUT}" "\\|" "append"

SG_PRECISION_TRAIN_TARGETS_PREDICHOS="${DIR_SUBGRUPOS}SG_${SG_ANALIZADO}/intermedio.csv.train_t_predicho.csv"
echo -e "<br>Dataset TRAIN - Targets predichos: (prediccion frozada e inútil) <a href=\"${SG_PRECISION_TRAIN_TARGETS_PREDICHOS}\">${SG_PRECISION_TRAIN_TARGETS_PREDICHOS}</a> --> "$(wc -l $SG_PRECISION_TRAIN_TARGETS_PREDICHOS | cut -d\  -f 1)" filas ("$(cat $SG_PRECISION_TRAIN_TARGETS_PREDICHOS  | grep 'True' |wc -l)" positivos)<br>" >> ${INFORME_OUT}

echo -e "<br>Ejemplos <b>(train - targets predichos)</b>:<br><br>" >> ${INFORME_OUT}
head -n 10 ${SG_PRECISION_TRAIN_TARGETS_PREDICHOS}  > "/tmp/entrada.csv"
java -jar ${PATH_JAR} --class "coordinador.Principal" "testIntegracion.ParserCsvEnTablaHtml" "/tmp/entrada.csv" "${INFORME_OUT}" "\\|" "append"



#####
SG_PRECISION_TEST_TARGETS_REALES="${DIR_SUBGRUPOS}SG_${SG_ANALIZADO}/intermedio.csv.ds_test_t.csv"
echo -e "<br>Dataset TEST - Targets reales: <a href=\"${SG_PRECISION_TEST_TARGETS_REALES}\">${SG_PRECISION_TEST_TARGETS_REALES}</a> --> "$(wc -l $SG_PRECISION_TEST_TARGETS_REALES | cut -d\  -f 1)" filas ("$(cat $SG_PRECISION_TEST_TARGETS_REALES  | grep 'True' |wc -l)" positivos)<br>" >> ${INFORME_OUT}

echo -e "<br>Ejemplos <b>(test - targets reales)</b>:<br><br>" >> ${INFORME_OUT}
head -n 10 ${SG_PRECISION_TEST_TARGETS_REALES}  > "/tmp/entrada.csv"
java -jar ${PATH_JAR} --class "coordinador.Principal" "testIntegracion.ParserCsvEnTablaHtml" "/tmp/entrada.csv" "${INFORME_OUT}" "\\|" "append"

SG_PRECISION_TEST_TARGETS_PREDICHOS="${DIR_SUBGRUPOS}SG_${SG_ANALIZADO}/intermedio.csv.test_t_predicho.csv"
echo -e "<br>Dataset TEST - Targets predichos: (prediccion frozada e inútil) <a href=\"${SG_PRECISION_TEST_TARGETS_PREDICHOS}\">${SG_PRECISION_TEST_TARGETS_PREDICHOS}</a> --> "$(wc -l $SG_PRECISION_TEST_TARGETS_PREDICHOS | cut -d\  -f 1)" filas ("$(cat $SG_PRECISION_TEST_TARGETS_PREDICHOS  | grep 'True' |wc -l)" positivos)<br>" >> ${INFORME_OUT}

echo -e "<br>Ejemplos <b>(test - targets predichos)</b>:<br><br>" >> ${INFORME_OUT}
head -n 10 ${SG_PRECISION_TEST_TARGETS_PREDICHOS}  > "/tmp/entrada.csv"
java -jar ${PATH_JAR} --class "coordinador.Principal" "testIntegracion.ParserCsvEnTablaHtml" "/tmp/entrada.csv" "${INFORME_OUT}" "\\|" "append"



#####
SG_PRECISION_V_TARGETS_REALES="${DIR_SUBGRUPOS}SG_${SG_ANALIZADO}/intermedio.csv.ds_validac_t.csv"
echo -e "<br>Dataset VALIDACION - Targets reales: <a href=\"${SG_PRECISION_V_TARGETS_REALES}\">${SG_PRECISION_V_TARGETS_REALES}</a> --> "$(wc -l $SG_PRECISION_V_TARGETS_REALES | cut -d\  -f 1)" filas ("$(cat $SG_PRECISION_V_TARGETS_REALES  | grep 'True' |wc -l)" positivos)<br>" >> ${INFORME_OUT}

echo -e "<br>Ejemplos <b>(validación - targets reales)</b>:<br><br>" >> ${INFORME_OUT}
head -n 10 ${SG_PRECISION_V_TARGETS_REALES}  > "/tmp/entrada.csv"
java -jar ${PATH_JAR} --class "coordinador.Principal" "testIntegracion.ParserCsvEnTablaHtml" "/tmp/entrada.csv" "${INFORME_OUT}" "\\|" "append"

SG_PRECISION_V_TARGETS_PREDICHOS="${DIR_SUBGRUPOS}SG_${SG_ANALIZADO}/intermedio.csv.validac_t_predicho.csv"
echo -e "<br>Dataset VALIDACION - Targets predichos: (prediccion frozada e inútil) <a href=\"${SG_PRECISION_V_TARGETS_PREDICHOS}\">${SG_PRECISION_V_TARGETS_PREDICHOS}</a> --> "$(wc -l $SG_PRECISION_V_TARGETS_PREDICHOS | cut -d\  -f 1)" filas ("$(cat $SG_PRECISION_V_TARGETS_PREDICHOS  | grep 'True' |wc -l)" positivos)<br>" >> ${INFORME_OUT}

echo -e "<br>Ejemplos <b>(validación - targets predichos)</b>:<br><br>" >> ${INFORME_OUT}
head -n 10 ${SG_PRECISION_V_TARGETS_PREDICHOS}  > "/tmp/entrada.csv"
java -jar ${PATH_JAR} --class "coordinador.Principal" "testIntegracion.ParserCsvEnTablaHtml" "/tmp/entrada.csv" "${INFORME_OUT}" "\\|" "append"



#####
file_log_pasado_mas_reciente=$(ls -Art /bolsa/logs/ | grep pasado | tail -n 1)
echo -e "<br><br>El fichero de LOG más reciente del pasado es:  <b>/bolsa/logs/${file_log_pasado_mas_reciente}</b><br>" >> ${INFORME_OUT}
echo -e "<br>La métrica mostrada en ese fichero de log para los datasets de train, test y validación es:<br><br>" >> ${INFORME_OUT}
cat "/bolsa/logs/${file_log_pasado_mas_reciente}" | grep reales | grep predichos | grep SG_${SG_ANALIZADO} | grep 'TRAIN' >> ${INFORME_OUT}
echo -e "<br>" >> ${INFORME_OUT}
cat "/bolsa/logs/${file_log_pasado_mas_reciente}" | grep reales | grep predichos | grep SG_${SG_ANALIZADO} | grep 'TEST' >> ${INFORME_OUT}
echo -e "<br>" >> ${INFORME_OUT}
cat "/bolsa/logs/${file_log_pasado_mas_reciente}" | grep reales | grep predichos | grep SG_${SG_ANALIZADO} | grep 'VALID' >> ${INFORME_OUT}
echo -e "<br>" >> ${INFORME_OUT}
echo -e "<h3>¡¡¡ Un sistema aleatorio/tonto acertaría un 50% de los casos, simplemente diciendo siempre false (o true) !!!</h3>" >> ${INFORME_OUT}
echo -e "<h3>RECORDAR: Solo miramos la precisión sobre los positivos predichos, porque es donde ponemos el DINERO REAL. No miramos los positivos NO predichos ni los negativos predichos.</h3>" >> ${INFORME_OUT}

echo -e "<br><br>" >> ${INFORME_OUT}



#######################################################################################################
#######################################################################################################
echo -e "<h2>******************* COMPROBACIONES del FUTURO2 ********************</h2>" >> ${INFORME_OUT}
echo -e "<br>" >> ${INFORME_OUT}

#####
echo -e "<br><h3>Capa 1.1 (brutos desestructurados)</h3>" >> ${INFORME_OUT}
BRUTO_YF="/bolsa/futuro/brutos/YF_NASDAQ_${empresa}.txt"
BRUTO_FZ="/bolsa/futuro/brutos/FZ_NASDAQ_${empresa}.html"
echo -e "Bruto - YahooFinance: <a href=\"${BRUTO_YF}\">${BRUTO_YF}</a> --> Tamanio (bytes) = "$(stat -c%s "$BRUTO_YF") >> ${INFORME_OUT}
echo -e "<br>Bruto - Finviz - Datos estáticos: <a href=\"${BRUTO_FZ}\">${BRUTO_FZ}</a> --> Tamanio (bytes) = "$(stat -c%s "$BRUTO_FZ") >> ${INFORME_OUT}

#####
echo -e "<br><h3>Capa 1.2 (brutos estructurados)</h3>" >> ${INFORME_OUT}
BRUTO_CSV="/bolsa/futuro/brutos_csv/NASDAQ_${empresa}.csv"
echo -e "Bruto CSV: <a href=\"${BRUTO_CSV}\">${BRUTO_CSV}</a> --> Tamanio (bytes) = "$(stat -c%s "$BRUTO_CSV")" con "$(wc -l $BRUTO_CSV | cut -d\  -f 1)" filas<br><br>" >> ${INFORME_OUT}
head -n 10 ${BRUTO_CSV} > "/tmp/entrada.csv"
java -jar ${PATH_JAR} --class "coordinador.Principal" "testIntegracion.ParserCsvEnTablaHtml" "/tmp/entrada.csv" "${INFORME_OUT}" "\\|" "append"

#####
echo -e "<br><h3>Capa 2 (limpios)</h3>" >> ${INFORME_OUT}
LIMPIO="/bolsa/futuro/limpios/NASDAQ_${empresa}.csv"
echo -e "Limpio: <a href=\"${LIMPIO}\">${LIMPIO}</a> --> Tamanio (bytes) = "$(stat -c%s "$LIMPIO")" con "$(wc -l $LIMPIO | cut -d\  -f 1)" filas<br><br>" >> ${INFORME_OUT}
head -n 5 ${LIMPIO}  > "/tmp/entrada.csv"
java -jar ${PATH_JAR} --class "coordinador.Principal" "testIntegracion.ParserCsvEnTablaHtml" "/tmp/entrada.csv" "${INFORME_OUT}" "\\|" "append"

#####
echo -e "<br><h3>Capa 3 (elaboradas)</h3>" >> ${INFORME_OUT}
ELABORADO="/bolsa/futuro/elaborados/NASDAQ_${empresa}.csv"
echo -e "Elaborado: <a href=\"${ELABORADO}\">${ELABORADO}</a> --> Tamanio (bytes) = "$(stat -c%s "$ELABORADO")" con "$(wc -l $ELABORADO | cut -d\  -f 1)" filas<br><br>" >> ${INFORME_OUT}
echo -e "Viendo la columna TARGET, este es el numero de casos de cada tipo: " >> ${INFORME_OUT}
echo -e "--> NULL = "$(cat /bolsa/futuro/elaborados/NASDAQ_AMRS.csv | grep '|null$' | wc -l) >> ${INFORME_OUT}
echo -e "  True ="$(cat /bolsa/futuro/elaborados/NASDAQ_AMRS.csv | grep '|1$' | wc -l) >> ${INFORME_OUT}
echo -e "  False = "$(cat /bolsa/futuro/elaborados/NASDAQ_AMRS.csv | grep '|0$' | wc -l) >> ${INFORME_OUT}
head -n 10 ${ELABORADO}  > "/tmp/entrada.csv"
java -jar ${PATH_JAR} --class "coordinador.Principal" "testIntegracion.ParserCsvEnTablaHtml" "/tmp/entrada.csv" "${INFORME_OUT}" "\\|" "append"

#####
echo -e "<br><h3>Capa 4 (subgrupos)</h3>" >> ${INFORME_OUT}
DIR_SUBGRUPOS="/bolsa/futuro/subgrupos/"
buscarEmpresaEnSubgrupos "${DIR_SUBGRUPOS}" "${empresa}" "${INFORME_OUT}"


#####
echo -e "<br><h3>Capa 5 (reducir, normalizar, etc) ...</h3><br>" >> ${INFORME_OUT}


#####
echo -e "<br><h3>Capa 6 (PREDICCION)</h3><br>" >> ${INFORME_OUT}

SG_TARGETS_PREDICHOS="${DIR_SUBGRUPOS}SG_${SG_ANALIZADO}/TARGETS_PREDICHOS.csv_humano"
echo -e "<br>Dataset FUTURO - Targets predichos: <a href=\"${SG_TARGETS_PREDICHOS}\">${SG_TARGETS_PREDICHOS}</a> --> "$(wc -l $SG_TARGETS_PREDICHOS | cut -d\  -f 1)" filas ("$(cat $SG_TARGETS_PREDICHOS  | grep 'True' |wc -l)" positivos)<br>" >> ${INFORME_OUT}

echo -e "<br>Ejemplos <b>(train - targets predichos)</b>:<br><br>" >> ${INFORME_OUT}
head -n 10 ${SG_TARGETS_PREDICHOS}  > "/tmp/entrada.csv"
java -jar ${PATH_JAR} --class "coordinador.Principal" "testIntegracion.ParserCsvEnTablaHtml" "/tmp/entrada.csv" "${INFORME_OUT}" "\\|" "append"



echo -e "</body></html>"  >> ${INFORME_OUT}
echo -e "TEST INTEGRACION - FIN: "$( date "+%Y%m%d%H%M%S" )

echo -e "<br><br>" >> ${INFORME_OUT}
