#!/bin/bash

echo -e $( date '+%Y%m%d_%H%M%S' )"20211130-PRINCIPAL - inicio"

############### OBTENER FICHERO CALIDAD.CSV #################

# Manualmente: en el fichero CONFIG hay que poner que entrene con 800 empresas en el pasado y 200 en el futuro !!!!!!!

# Entrena modelo con datos de hace 50 días y luego itera hasta hoy. Con ello pinta las gráficas y calcula el CALIDAD.CSV
#/home/carloslinux/Desktop/GIT_BOLSA/bolsa/BolsaScripts/modelofuturosimpleyanalisis.sh

FILE=/home/carloslinux/Dropbox/BOLSA_PREDICTOR/ANALISIS/CALIDAD.csv
if [ -f "$FILE" ]; then
    echo "El fichero $FILE existe."
else 
    echo "El fichero $FILE no existe!!!"
	echo "Hay algo mal. Saliendo..."
	exit -1
fi


#################### DESCRIPCIÓN Y PARÁMETROS FUNDAMENTALES ###############################
# Entrena modelos (pasado) para generar los ficheros de realimentacion/ (estadisticas de falsos positivos)
# Calcula los estadísticas de falsos positivos y coloca los ficheros CSV en la carpeta realimentacion/ (que es permanente, no volátil)
# Otra vez, entrena modelos con el pasado (aplicando la realimentacion/ aprendida).

#DESCOMENTAR --> echo -e $( date '+%Y%m%d_%H%M%S' )"20211130-PRINCIPAL - entrenarSoloPasado.sh (RONDA 1 para tener datos con los que calcular falsos positivos)"
#DESCOMENTAR --> /home/carloslinux/Desktop/GIT_BOLSA/bolsa/BolsaScripts/entrenarSoloPasado.sh

#DESCOMENTAR --> echo -e $( date '+%Y%m%d_%H%M%S' )"20211130-PRINCIPAL - AnalisisFalsosPositivos.py"
#DESCOMENTAR --> /home/carloslinux/anaconda3/envs/BolsaPython38/bin/python /home/carloslinux/Desktop/GIT_BOLSA/bolsa/BolsaPython/bolsa/AnalisisFalsosPositivos.py /home/carloslinux/Desktop/GIT_BOLSA/bolsa/BolsaJava/realimentacion/

echo -e $( date '+%Y%m%d_%H%M%S' )"20211130-PRINCIPAL - entrenarSoloPasadoYpredecirSoloFut1.sh (RONDA 2: la REALIMENTACION de falsos positivos esta a SI)"
/home/carloslinux/Desktop/GIT_BOLSA/bolsa/BolsaScripts/entrenarSoloPasadoYpredecirSoloFut1.sh
################################################################################################


echo -e $( date '+%Y%m%d_%H%M%S' )"20211130-PRINCIPAL - fin"


