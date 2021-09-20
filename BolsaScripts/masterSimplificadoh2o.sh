#!/bin/bash

#set -e

echo -e "MASTER - INICIO: "$( date "+%Y%m%d%H%M%S" )

#################### DIRECTORIOS ###############################################################
DIR_BASE="/bolsa/"
DIR_CODIGOS_CARLOS="/home/carloslinux/Desktop/GIT_BOLSA/"
DIR_CODIGOS_LUIS="/home/t151521${DIR_BASE}"
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

crearCarpetaSiNoExistePeroNoVaciar() {
	param1=${1} 			#directorio
	echo "Creando carpeta: $param1"
	mkdir -p ${param1}
	chmod 777 -Rf ${param1}
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

DIR_LOGS="${DIR_BASE}logs/"
DIR_BRUTOS="${DIR_BASE}${DIR_TIEMPO}/brutos/"
DIR_BRUTOS_CSV="${DIR_BASE}${DIR_TIEMPO}/brutos_csv/"
DIR_LIMPIOS="${DIR_BASE}${DIR_TIEMPO}/limpios/"
DIR_ELABORADOS="${DIR_BASE}${DIR_TIEMPO}/elaborados/"
DIR_SUBGRUPOS="${DIR_BASE}${DIR_TIEMPO}/subgrupos/"
DIR_IMG="img/"
DIR_TRAMIF="tramif/"

crearCarpetaSiNoExiste "${DIR_LOGS}"
# AQUÍ NO SE VACIARÁ LA CARPETA SI YA EXISTE
crearCarpetaSiNoExistePeroNoVaciar "${DIR_BASE}${DIR_TIEMPO}"

############### LOGS ########################################################
rm -f "${DIR_LOGS}log4j.log"
LOG_MASTER="${DIR_LOGS}${ID_EJECUCION}_bolsa_coordinador_${DIR_TIEMPO}.log"
rm -f "${LOG_MASTER}"

############### COMPILAR JAR ########################################################
echo -e "Compilando JAVA en un JAR..." >> ${LOG_MASTER}
cd "${DIR_JAVA}" >> ${LOG_MASTER}
rm -Rf "${DIR_JAVA}target/" >> ${LOG_MASTER}
mvn clean compile assembly:single >> ${LOG_MASTER}


if [ "$ACTIVAR_SG_Y_PREDICCION" = "S" ];  then

	############  PARA CADA SUBGRUPO ###############################################################
	
	for i in $(seq 0 100)
	do
		dir_subgrupo="${DIR_SUBGRUPOS}SG_${i}/"
		
		if [ -d "$dir_subgrupo" ]; then  # Comprueba si existe el directorio
		
			echo "\n\n\n"$( date '+%Y%m%d_%H%M%S' )" -------- Subgrupo cuya carpeta es: ${dir_subgrupo} --------" >> ${LOG_MASTER}
			
			path_dir_pasado=$( echo ${dir_subgrupo} | sed "s/futuro/pasado/" )
			echo "$path_dir_pasado"
			path_leaderboard="${path_dir_pasado}aml_leaderboard.h2o"
			echo "$path_leaderboard"
		
			if [ "$DIR_TIEMPO" = "pasado" ] || { [ "$DIR_TIEMPO" = "futuro" ] && [ -f "$path_leaderboard" ]; };  then 
			
				crearCarpetaSiNoExisteYVaciar  "${dir_subgrupo}${DIR_IMG}"
				crearCarpetaSiNoExisteYVaciar  "${dir_subgrupo}${DIR_TRAMIF}"
				
				echo -e $( date '+%Y%m%d_%H%M%S' )" ##################### Capa 5 y 6 con H2O #####################" >> ${LOG_MASTER}
				echo -e $( date '+%Y%m%d_%H%M%S' )" PASADO ó FUTURO: Se aplica AutoML de H2O para generar el mejor modelo de cada subgrupo. Guarda el modelo GANADOR de cada subgrupo..." >> ${LOG_MASTER}
				$PYTHON_MOTOR "${PYTHON_SCRIPTS}bolsa/C5C6H2O.py" "${dir_subgrupo}/" "${DIR_TIEMPO}" "${DESPLAZAMIENTO_ANTIGUEDAD}"  >> ${LOG_MASTER}
				
			else
				echo "Al evaluar el subgrupo cuyo directorio es $dir_subgrupo para el tiempo $DIR_TIEMPO vemos que no existe entrenamiento en el pasado, asi que no existe $path_leaderboard" >> ${LOG_MASTER}
			fi
		fi
	done

fi

################################################################################################
echo -e "Actualizando informe HTML de uso de DATOS..." >> ${LOG_MASTER}
java -jar ${PATH_JAR} --class "coordinador.Principal" "c70X.validacion.GeneradorInformeHtml" "${DIR_BASE}${DIR_TIEMPO}/" 2>>${LOG_MASTER} 1>>${LOG_MASTER}

echo -e "Actualizando informe HTML de uso de FEATURES (mirando el pasado)..." >> ${LOG_MASTER}
DIR_SUBGRUPOS_PASADO=$(echo ${DIR_SUBGRUPOS} | sed -e "s/futuro/pasado/g")
$PYTHON_MOTOR "${PYTHON_SCRIPTS}bolsa/FeaturesAnalisisPosteriori.py" "${DIR_SUBGRUPOS_PASADO}" "${DIR_BASE}pasado/matriz_features_antes_de_pca.html" "${DIR_BASE}pasado/matriz_features.html" 2>>${LOG_MASTER} 1>>${LOG_MASTER}


echo -e "MASTER - FIN: "$( date "+%Y%m%d%H%M%S" )
echo -e "******** FIN de master**************" >> ${LOG_MASTER}


