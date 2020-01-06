#!/bin/bash

echo -e "MASTER - INICIO: "$( date "+%Y%m%d%H%M%S" )

################## PARAMETROS DE ENTRADA ############################################
DIR_TIEMPO="${1}" #pasado o futuro
DESPLAZAMIENTO_ANTIGUEDAD="${2}"  #0, 50 ....
ES_ENTORNO_VALIDACION="${3}" #0, 1
ACTIVAR_DESCARGA="${4}" #S o N

# PARAMETROS DE TARGET MEDIDOS EN VELAS
S="${5}" #default=10
X="${6}" #default=56
R="${7}" #default=10
M="${8}" #default=7
F="${9}" #default=5


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

################ VARIABLES DE EJECUCION #########################################################
ID_EJECUCION=$( date "+%Y%m%d%H%M%S" )
echo -e "ID_EJECUCION = "${ID_EJECUCION}

NUM_MAX_EMPRESAS_DESCARGADAS=300
MIN_COBERTURA_CLUSTER=60
MIN_EMPRESAS_POR_CLUSTER=10

#################### DIRECTORIOS ###############################################################
DIR_CODIGOS_CARLOS="/home/carloslinux/Desktop/GIT_BOLSA/" #NO TOCAR
DIR_CODIGOS_LUIS="/home/t151521/bolsa/" #NO TOCAR
DIR_CODIGOS="${DIR_CODIGOS_CARLOS}"

PATH_SCRIPTS="${DIR_CODIGOS}BolsaScripts/"
PYTHON_SCRIPTS="${DIR_CODIGOS}BolsaPython/"
DIR_JAVA="${DIR_CODIGOS}BolsaJava/"
PATH_JAR="${DIR_JAVA}target/bolsajava-1.0-jar-with-dependencies.jar"
PYTHON_MOTOR="/home/carloslinux/Desktop/PROGRAMAS/anaconda3/envs/BolsaPython/bin/python"

DIR_BASE="/bolsa/"
DIR_LOGS="${DIR_BASE}logs/"
DIR_BRUTOS="${DIR_BASE}${DIR_TIEMPO}/brutos/"
DIR_BRUTOS_CSV="${DIR_BASE}${DIR_TIEMPO}/brutos_csv/"
DIR_LIMPIOS="${DIR_BASE}${DIR_TIEMPO}/limpios/"
DIR_ELABORADOS="${DIR_BASE}${DIR_TIEMPO}/elaborados/"
DIR_SUBGRUPOS="${DIR_BASE}${DIR_TIEMPO}/subgrupos/"
DIR_IMG="img/"

crearCarpetaSiNoExiste "${DIR_LOGS}"
crearCarpetaSiNoExisteYVaciar "${DIR_BASE}${DIR_TIEMPO}"

############### LOGS ########################################################
rm -f "${DIR_LOGS}log4j.log"
LOG_MASTER="${DIR_LOGS}${ID_EJECUCION}_bolsa_coordinador_${DIR_TIEMPO}.log"
rm -f "${LOG_MASTER}"

############### COMPILAR JAR ########################################################
echo -e "Compilando JAVA en un JAR..." >> ${LOG_MASTER}
cd "${DIR_JAVA}" >> ${LOG_MASTER}
rm -Rf "${DIR_JAVA}target/" >> ${LOG_MASTER}
mvn clean compile assembly:single >> ${LOG_MASTER}

################################################################################################
echo -e "-------- DATOS BRUTOS -------------" >> ${LOG_MASTER}

if [ "$ACTIVAR_DESCARGA" = "S" ];  then

	crearCarpetaSiNoExisteYVaciar "${DIR_BRUTOS}"
	crearCarpetaSiNoExisteYVaciar "${DIR_BRUTOS_CSV}"

	echo -e "DINAMICOS - Descargando de YAHOO FINANCE..." >> ${LOG_MASTER}
	java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c10X.brutos.YahooFinance01Descargar" "${NUM_MAX_EMPRESAS_DESCARGADAS}" "${DIR_BRUTOS}" "${DIR_TIEMPO}" "${ES_ENTORNO_VALIDACION}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}

	echo -e "DINAMICOS - Limpieza de YAHOO FINANCE..." >> ${LOG_MASTER}
	java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c10X.brutos.YahooFinance02Parsear" "${DIR_BRUTOS}" "${DIR_BRUTOS_CSV}" "${DIR_TIEMPO}" "${ES_ENTORNO_VALIDACION}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}

	echo -e "ESTATICOS - Descargando de FINVIZ (igual para Pasado o Futuro, salvo el directorio)..." >> ${LOG_MASTER}
	java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c10X.brutos.EstaticosFinvizDescargarYParsear" "${NUM_MAX_EMPRESAS_DESCARGADAS}" "${DIR_BRUTOS}" "${DIR_BRUTOS_CSV}" "${ES_ENTORNO_VALIDACION}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}

	echo -e "ESTATICOS + DINAMICOS: juntando en un CSV único..." >> ${LOG_MASTER}
	java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c10X.brutos.JuntarEstaticosYDinamicosCSVunico" "${DIR_BRUTOS_CSV}" "${DESPLAZAMIENTO_ANTIGUEDAD}" "${ES_ENTORNO_VALIDACION}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}

	echo -e "ESTATICOS + DINAMICOS: limpiando CSVs intermedios brutos..." >> ${LOG_MASTER}
	java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c10X.brutos.LimpiarCSVBrutosTemporales" "${DIR_BRUTOS_CSV}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}

fi;

NUM_FICHEROS_10x=$(ls -l ${DIR_BRUTOS_CSV} | wc -l)
echo -e "La capa 10X ha generado $NUM_FICHEROS_10x ficheros" 2>>${LOG_MASTER} 1>>${LOG_MASTER}
if [ "$NUM_FICHEROS_10x" -lt 1 ]; then
	echo -e "El numero de ficheros es menor que el de la capa anterior. Asi que se han perdido algunos por algun problema. Debemos analizarlo. Saliendo..." 2>>${LOG_MASTER} 1>>${LOG_MASTER}
	exit -1
fi

################################################################################################
echo -e "-------- DATOS LIMPIOS -------------" >> ${LOG_MASTER}

#######echo -e "Operaciones de limpieza: quitar outliers, rellenar missing values..." >> ${LOG_MASTER}
#######java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c30X.elaborados.LimpiarOperaciones" "${DIR_BRUTOS_CSV}" "${DIR_LIMPIOS}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}

crearCarpetaSiNoExisteYVaciar "${DIR_LIMPIOS}"

cp ${DIR_BRUTOS_CSV}*.csv ${DIR_LIMPIOS} 2>>${LOG_MASTER} 1>>${LOG_MASTER}

NUM_FICHEROS_20x=$(ls -l ${DIR_LIMPIOS} | wc -l)
echo -e "La capa 20X ha generado $NUM_FICHEROS_20x ficheros" 2>>${LOG_MASTER} 1>>${LOG_MASTER}
if [ "$NUM_FICHEROS_20x" -lt "$NUM_FICHEROS_10x" ]; then
	echo -e "El numero de ficheros es menor que el de la capa anterior. Asi que se han perdido algunos por algun problema. Debemos analizarlo. Saliendo..." 2>>${LOG_MASTER} 1>>${LOG_MASTER}
	exit -1
fi

################################################################################################
echo -e "-------- VARIABLES ELABORADAS -------------" >> ${LOG_MASTER}

crearCarpetaSiNoExisteYVaciar "${DIR_ELABORADOS}"

echo -e "Calculando elaborados y target..." >> ${LOG_MASTER}
java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c30X.elaborados.ConstructorElaborados" "${DIR_LIMPIOS}" "${DIR_ELABORADOS}" "${S}" "${X}" "${R}" "${M}" "${F}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}

echo -e "Elaborados (incluye la variable elaborada TARGET) ya calculados" >> ${LOG_MASTER}

NUM_FICHEROS_30x=$(ls -l ${DIR_ELABORADOS} | wc -l)
echo -e "La capa 30X ha generado $NUM_FICHEROS_30x ficheros" 2>>${LOG_MASTER} 1>>${LOG_MASTER}
if [ "$NUM_FICHEROS_30x" -lt "$NUM_FICHEROS_20x" ]; then
	echo -e "El numero de ficheros es menor que el de la capa anterior. Asi que se han perdido algunos por algun problema. Debemos analizarlo. Saliendo..." 2>>${LOG_MASTER} 1>>${LOG_MASTER}
	exit -1
fi

################################################################################################
echo -e "-------- SUBGRUPOS -------------" >> ${LOG_MASTER}

crearCarpetaSiNoExisteYVaciarRecursivo "${DIR_SUBGRUPOS}"

java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c40X.subgrupos.CrearDatasetsSubgrupos" "${DIR_ELABORADOS}" "${DIR_SUBGRUPOS}" "${MIN_COBERTURA_CLUSTER}" "${MIN_EMPRESAS_POR_CLUSTER}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}
#java -Djava.util.logging.SimpleFormatter.format="%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$s %2$s %5$s%6$s%n" -jar ${PATH_JAR} --class "coordinador.Principal" "c40X.subgrupos.CrearDatasetsSubgruposKMeans" "${DIR_ELABORADOS}" "${DIR_SUBGRUPOS}" "${MIN_COBERTURA_CLUSTER}" "${MIN_EMPRESAS_POR_CLUSTER}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}

############  PARA CADA SUBGRUPO ###############################################################

for dir_subgrupo in ${DIR_SUBGRUPOS}*/
do
	echo "-------- Subgrupo cuya carpeta es: ${dir_subgrupo} --------" >> ${LOG_MASTER}
	
	path_dir_pasado=$( echo ${dir_subgrupo} | sed "s/futuro/pasado/" )
	echo "$path_dir_pasado"
	path_normalizador_pasado="${path_dir_pasado}NORMALIZADOR.tool"
	echo "$path_normalizador_pasado"
	
	if [ "$DIR_TIEMPO" = "pasado" ] || { [ "$DIR_TIEMPO" = "futuro" ] && [ -f "$path_normalizador_pasado" ]; };  then 
        
		crearCarpetaSiNoExisteYVaciar  "${dir_subgrupo}${DIR_IMG}"
	
	    echo -e "Capa 5: elimina MISSING VALUES (NA en columnas y filas), elimina OUTLIERS, balancea clases (undersampling de mayoritaria), calcula IMG funciones de densidad, NORMALIZA las features, comprueba suficientes casos en clase minoritaria, REDUCCION de FEATURES y guarda el CSV REDUCIDO..." >> ${LOG_MASTER}
	    $PYTHON_MOTOR "${PYTHON_SCRIPTS}bolsa/C5NormalizarYReducirDatasetSubgrupo.py" "${dir_subgrupo}/" "${DIR_TIEMPO}" >> ${LOG_MASTER}
	    
	    echo -e "Capa 6 - PASADO ó FUTURO: balancea las clases (aunque ya se hizo en capa 5), divide dataset de entrada (entrenamiento, test, validación), CREA MODELOS (con hyperparámetros)  los evalúa. Guarda el modelo GANADOR de cada subgrupo..." >> ${LOG_MASTER}
	    $PYTHON_MOTOR "${PYTHON_SCRIPTS}bolsa/C6CreadorModelosDeSubgrupo.py" "${dir_subgrupo}/" "${DIR_TIEMPO}" "${DESPLAZAMIENTO_ANTIGUEDAD}"  >> ${LOG_MASTER}
		
    else
        echo "Al evaluar el subgrupo cuyo directorio es $dir_subgrupo para el tiempo $DIR_TIEMPO vemos que no existe entrenamiento en el pasado, asi que no existe $path_normalizador_pasado" >> ${LOG_MASTER}
    fi;
	
done

################################################################################################
echo -e "MASTER - FIN: "$( date "+%Y%m%d%H%M%S" )
echo -e "******** FIN de master**************" >> ${LOG_MASTER}
