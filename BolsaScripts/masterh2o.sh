#!/bin/bash

#set -e

echo -e "MASTER - INICIO: "$( date "+%Y%m%d%H%M%S" )

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

################## PARAMETROS DE ENTRADA ############################################
DIR_TIEMPO="${1}" #pasado o futuro
DESPLAZAMIENTO_ANTIGUEDAD="${2}"  #0, 50 ....
ES_ENTORNO_VALIDACION="${3}" #0, 1
ACTIVAR_DESCARGA="${4}" #S o N
ACTIVAR_SG_Y_PREDICCION="${5}" #S o N

# PARAMETROS DE TARGET MEDIDOS EN VELAS
S="${6}" #default=10
X="${7}" #default=56
R="${8}" #default=10
M="${9}" #default=7
F="${10}" #default=5
B="${11}" #default=5
NUM_MAX_EMPRESAS_DESCARGADAS="${12}"  #default=100
UMBRAL_SUBIDA_POR_VELA="${13}"  #default=3
UMBRAL_MINIMO_GRAN_VELA="${14}"  #default=9999
MIN_COBERTURA_CLUSTER="${15}"  #Porcentaje de empresas con al menos una vela positiva
MIN_EMPRESAS_POR_CLUSTER="${16}"
P_INICIO="${17}" #Periodo de entrenamiento (inicio)
P_FIN="${18}" #Periodo de entrenamiento (fin)
MAX_NUM_FEAT_REDUCIDAS="${19}"
CAPA5_MAX_FILAS_ENTRADA="${20}"  # Maximo numero de filas permitido en entrada a capa 5 (por rendimiento)
DINAMICA1="${21}" # Default = 1
DINAMICA2="${22}" # Default = 1


################## FUNCIONES #############################################################
crearCarpetaSiNoExiste() {
	param1=${1} 			#directorio
	echo "Creando carpeta: $param1"
	mkdir -p ${param1}
	chmod 777 -Rf ${param1}
}

crearCarpetaSiNoExisteYVaciar() {
	param1=${1} 			#directorio
	echo "Creando carpeta: $param1"
	mkdir -p ${param1}
	chmod 777 -Rf ${param1}
	rm -f ${param1}*
}

crearCarpetaSiNoExisteYVaciarRecursivo() {
	param1=${1} 			#directorio
	echo "Creando carpeta: $param1"
	mkdir -p ${param1}
	chmod 777 -Rf ${param1}
	rm -Rf ${param1}*
}

comprobarQueDirectorioNoEstaVacio(){
	param1="${1}"
	a=`(ls -lrt "${param1}"  | grep "total 0")`
	if [ $? == "0" ]; then
		echo "El directorio está vacio, pero debería estar lleno!!! DIR: ${param1}   Saliendo..."
		exit -1
	fi
}

################ VARIABLES DE EJECUCION #########################################################
ID_EJECUCION=$( date "+%Y%m%d%H%M%S" )
echo -e "ID_EJECUCION = "${ID_EJECUCION}

PATH_SCRIPTS="${DIR_CODIGOS}BolsaScripts/"
PYTHON_SCRIPTS="${DIR_CODIGOS}BolsaPython/"
DIR_JAVA="${DIR_CODIGOS}BolsaJava/"
PATH_JAR="${DIR_JAVA}target/bolsajava-1.0-jar-with-dependencies.jar"

DIR_BASE="/bolsa/"
DIR_LOGS="${DIR_BASE}logs/"
DIR_BRUTOS="${DIR_BASE}${DIR_TIEMPO}/brutos/"
DIR_BRUTOS_CSV="${DIR_BASE}${DIR_TIEMPO}/brutos_csv/"
DIR_LIMPIOS="${DIR_BASE}${DIR_TIEMPO}/limpios/"
DIR_ELABORADOS="${DIR_BASE}${DIR_TIEMPO}/elaborados/"
DIR_SUBGRUPOS="${DIR_BASE}${DIR_TIEMPO}/subgrupos/"
DIR_IMG="img/"
DIR_TRAMIF="tramif/"

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
echo -e $( date '+%Y%m%d_%H%M%S' )" -------- Capa 1: DATOS BRUTOS -------------" >> ${LOG_MASTER}

if [ "${ACTIVAR_DESCARGA}" = "S" ];  then

	crearCarpetaSiNoExisteYVaciar "${DIR_BRUTOS}"
	crearCarpetaSiNoExisteYVaciar "${DIR_BRUTOS_CSV}"

	echo -e "DINAMICOS - Descargando de YAHOO FINANCE..." >> ${LOG_MASTER}
	java -jar ${PATH_JAR} --class "coordinador.Principal" "c10X.brutos.YahooFinance01Descargar" "${NUM_MAX_EMPRESAS_DESCARGADAS}" "${DIR_BRUTOS}" "${DIR_TIEMPO}" "${RANGO_YF}" "${VELA_YF}" "${ES_ENTORNO_VALIDACION}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}

	echo -e "DINAMICOS - Limpieza de YAHOO FINANCE..." >> ${LOG_MASTER}
	java -jar ${PATH_JAR} --class "coordinador.Principal" "c10X.brutos.YahooFinance02Parsear" "${DIR_BRUTOS}" "${DIR_BRUTOS_CSV}" "${DIR_TIEMPO}" "${ES_ENTORNO_VALIDACION}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}
	
	#PENDIENTE DINAMICOS: YAHOO FINANCE (solo modo FUTURO) --> https://finance.yahoo.com/quote/AAPL?p=AAPL --> Campo: Earnings Date
	#PENDIENTE DINAMICOS: NASDAQOLD_EARNINGS (solo modo PASADO) --> https://old.nasdaq.com/symbol/aapl/earnings-surprise --> tabla de fechas

	echo -e "ESTATICOS - Descargando de FINVIZ (igual para Pasado o Futuro, salvo el directorio)..." >> ${LOG_MASTER}
	java -jar ${PATH_JAR} --class "coordinador.Principal" "c10X.brutos.EstaticosFinvizDescargarYParsear" "${NUM_MAX_EMPRESAS_DESCARGADAS}" "${DIR_BRUTOS}" "${DIR_BRUTOS_CSV}" "${ES_ENTORNO_VALIDACION}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}

	echo -e "ESTATICOS + DINAMICOS: juntando en un CSV único..." >> ${LOG_MASTER}
	java -jar ${PATH_JAR} --class "coordinador.Principal" "c10X.brutos.JuntarEstaticosYDinamicosCSVunico" "${DIR_BRUTOS_CSV}" "${DESPLAZAMIENTO_ANTIGUEDAD}" "${ES_ENTORNO_VALIDACION}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}

	echo -e "ESTATICOS + DINAMICOS: limpiando CSVs intermedios brutos..." >> ${LOG_MASTER}
	java -jar ${PATH_JAR} --class "coordinador.Principal" "c10X.brutos.LimpiarCSVBrutosTemporales" "${DIR_BRUTOS_CSV}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}

fi;

comprobarQueDirectorioNoEstaVacio "${DIR_BRUTOS}"
comprobarQueDirectorioNoEstaVacio "${DIR_BRUTOS_CSV}"

#Borramos fichero auxiliar de velas
echo "Borrando: ${DIR_BRUTOS_CSV}/VELAS*" 2>>${LOG_MASTER} 1>>${LOG_MASTER}
rm ${DIR_BRUTOS_CSV}/VELAS* 2>>${LOG_MASTER} 1>>${LOG_MASTER}

NUM_FICHEROS_10x=$(ls -l ${DIR_BRUTOS_CSV} | grep -v 'total' | wc -l)
echo -e "La capa 10X ha generado $NUM_FICHEROS_10x ficheros" 2>>${LOG_MASTER} 1>>${LOG_MASTER}
if [ "$NUM_FICHEROS_10x" -lt 1 ]; then
	echo -e "El numero de ficheros es 0. Asi que se han perdido algunos por algun problema. Debemos analizarlo. Saliendo..." 2>>${LOG_MASTER} 1>>${LOG_MASTER}
	exit -1
fi

################################################################################################
echo -e $( date '+%Y%m%d_%H%M%S' )" -------- Capa 2: DATOS LIMPIOS -------------" >> ${LOG_MASTER}

crearCarpetaSiNoExisteYVaciar "${DIR_LIMPIOS}"

echo -e "Operaciones de limpieza: limitar periodo..." >> ${LOG_MASTER}
java -jar ${PATH_JAR} --class "coordinador.Principal" "c30X.elaborados.LimpiarOperaciones" "${DIR_BRUTOS_CSV}" "${DIR_LIMPIOS}" "${P_INICIO}" "${P_FIN}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}

# cp ${DIR_BRUTOS_CSV}*.csv ${DIR_LIMPIOS} 2>>${LOG_MASTER} 1>>${LOG_MASTER}

NUM_FICHEROS_20x=$(ls -l ${DIR_LIMPIOS} | grep -v 'total' | wc -l)
echo -e "La capa 20X ha generado $NUM_FICHEROS_20x ficheros" 2>>${LOG_MASTER} 1>>${LOG_MASTER}
if [ "$NUM_FICHEROS_20x" -lt "$NUM_FICHEROS_10x" ]; then
	echo -e "El numero de ficheros es menor que el de la capa anterior. Asi que se han perdido algunos por algun problema. Debemos analizarlo. Saliendo..." 2>>${LOG_MASTER} 1>>${LOG_MASTER}
	exit -1
fi

comprobarQueDirectorioNoEstaVacio "${DIR_LIMPIOS}"

################################################################################################
echo -e $( date '+%Y%m%d_%H%M%S' )" -------- Capa 3: VARIABLES ELABORADAS -------------" >> ${LOG_MASTER}

crearCarpetaSiNoExisteYVaciar "${DIR_ELABORADOS}"

echo -e "Calculando elaborados y target..." >> ${LOG_MASTER}
java -jar ${PATH_JAR} --class "coordinador.Principal" "c30X.elaborados.ConstructorElaborados" "${DIR_LIMPIOS}" "${DIR_ELABORADOS}" "${S}" "${X}" "${R}" "${M}" "${F}" "${B}" "${UMBRAL_SUBIDA_POR_VELA}" "${UMBRAL_MINIMO_GRAN_VELA}" "${DINAMICA1}" "${DINAMICA2}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}

echo -e "Elaborados (incluye la variable elaborada TARGET) ya calculados" >> ${LOG_MASTER}

NUM_FICHEROS_30x=$(ls -l ${DIR_ELABORADOS} | grep -v 'total' | wc -l)
echo -e "La capa 30X ha generado $NUM_FICHEROS_30x ficheros" 2>>${LOG_MASTER} 1>>${LOG_MASTER}
if [ "$NUM_FICHEROS_30x" -lt "$NUM_FICHEROS_20x" ]; then
	echo -e "El numero de ficheros es menor que el de la capa anterior. Asi que se han perdido algunos por algun problema. Debemos analizarlo. Saliendo..." 2>>${LOG_MASTER} 1>>${LOG_MASTER}
	exit -1
fi

comprobarQueDirectorioNoEstaVacio "${DIR_ELABORADOS}"

############## Calcular Subgrupos ####################################################################

if [ "$ACTIVAR_SG_Y_PREDICCION" = "S" ];  then
	
	echo -e $( date '+%Y%m%d_%H%M%S' )" -------- Capa 4: SUBGRUPOS -------------" >> ${LOG_MASTER}
	crearCarpetaSiNoExisteYVaciarRecursivo "${DIR_SUBGRUPOS}"
	java -jar ${PATH_JAR} --class "coordinador.Principal" "c40X.subgrupos.CrearDatasetsSubgrupos" "${DIR_ELABORADOS}" "${DIR_SUBGRUPOS}" "${MIN_COBERTURA_CLUSTER}" "${MIN_EMPRESAS_POR_CLUSTER}" "${DIR_TIEMPO}" "${DINAMICA1}" "${DINAMICA2}" 2>>${LOG_MASTER} 1>>${LOG_MASTER}

	############  PARA CADA SUBGRUPO ###############################################################
	
	for i in $(seq 0 100)
	do
		dir_subgrupo="${DIR_SUBGRUPOS}SG_${i}/"
		
		if [ -d "$dir_subgrupo" ]; then  # Comprueba si existe el directorio
		
			echo "\n\n\n"$( date '+%Y%m%d_%H%M%S' )" -------- Subgrupo cuya carpeta es: ${dir_subgrupo} --------" >> ${LOG_MASTER}
			
			path_dir_pasado=$( echo ${dir_subgrupo} | sed "s/futuro/pasado/" )
			echo "$path_dir_pasado"
			
			crearCarpetaSiNoExisteYVaciar  "${dir_subgrupo}${DIR_IMG}"
			crearCarpetaSiNoExisteYVaciar  "${dir_subgrupo}${DIR_TRAMIF}"
				
			echo -e $( date '+%Y%m%d_%H%M%S' )" ##################### Capa 5 y 6 con H2O #####################" >> ${LOG_MASTER}
			echo -e $( date '+%Y%m%d_%H%M%S' )" PASADO ó FUTURO: Se aplica AutoML de H2O para generar el mejor modelo de cada subgrupo. Guarda el modelo GANADOR de cada subgrupo..." >> ${LOG_MASTER}
			$PYTHON_MOTOR "${PYTHON_SCRIPTS}bolsa/C5C6H2O.py" "${dir_subgrupo}/" "${DIR_TIEMPO}" "${DESPLAZAMIENTO_ANTIGUEDAD}"  >> ${LOG_MASTER}
				
		fi
	done

fi

################################################################################################
echo -e "Actualizando informe HTML de uso de DATOS..." >> ${LOG_MASTER}
java -jar ${PATH_JAR} --class "coordinador.Principal" "c70X.validacion.GeneradorInformeHtml" "${DIR_BASE}${DIR_TIEMPO}/" 2>>${LOG_MASTER} 1>>${LOG_MASTER}

echo -e "Actualizando informe HTML de uso de FEATURES (mirando el pasado)..." >> ${LOG_MASTER}
DIR_SUBGRUPOS_PASADO=$(echo ${DIR_SUBGRUPOS} | sed -e "s/futuro/pasado/g")
$PYTHON_MOTOR "${PYTHON_SCRIPTS}bolsa/FeaturesAnalisisPosteriori.py" "${DIR_SUBGRUPOS_PASADO}" "/bolsa/pasado/matriz_features_antes_de_pca.html" "/bolsa/pasado/matriz_features.html" 2>>${LOG_MASTER} 1>>${LOG_MASTER}


echo -e "MASTER - FIN: "$( date "+%Y%m%d%H%M%S" )
echo -e "******** FIN de master**************" >> ${LOG_MASTER}


