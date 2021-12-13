#!/bin/bash

###########################################
# Entrena un modelo en el pasado hace 50 días
# Itera desde el día con antigüedad 50 al 0  (no descarga datos)
# Predice el futuro en el día 0
# 
# Debemos poner estos numeros en el CONFIG:
# 			export NUM_EMPRESAS="1000"  # Numero de empresas descargadas (para ENTRENAR)
# 			export NUM_EMPRESAS_INVERSION="100"  #Numero de empresas descargadas (REAL, para invertir)
#########################################################

echo -e "INICIO: "$( date "+%Y%m%d%H%M%S" )

./generaModelo.sh
./inversionHistoricaSoloFuturoSimplificado.sh
./inversionAnalisis.sh

echo -e "FIN: "$( date "+%Y%m%d%H%M%S" )

