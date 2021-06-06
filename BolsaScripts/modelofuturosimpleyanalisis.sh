#!/bin/bash


echo -e "INICIO: "$( date "+%Y%m%d%H%M%S" )

./generaModelo.sh
./inversionHistoricaSoloFuturoSimplificado.sh
./inversionAnalisis.sh

echo -e "FIN: "$( date "+%Y%m%d%H%M%S" )

