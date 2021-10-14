#!/bin/bash


echo -e "INICIO: "$( date "+%Y%m%d%H%M%S" )

./generaModeloh2o.sh
./inversionHistoricaSoloFuturoSimplificadoh2o.sh
./inversionAnalisis.sh

echo -e "FIN: "$( date "+%Y%m%d%H%M%S" )

