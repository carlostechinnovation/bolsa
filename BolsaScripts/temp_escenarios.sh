#!/bin/bash

###########################################################################################################
#000 Ejecucion normal, para ACTUALIZAR los datos estáticos (con la descarga de datos activada) para 1300 empresas
/home/carloslinux/Desktop/GIT_BOLSA/BolsaScripts/validacionEuros.sh

rm -Rf /bolsa/validacion_datos/
mkdir -p /bolsa/validacion_datos/pasado_brutos_csv/
cp -a "/bolsa/validacion/"$(ls /bolsa/validacion/ | grep 'pasado')"/." "/bolsa/validacion_datos/pasado_brutos_csv/"
mkdir -p /bolsa/validacion_datos/futuro1_brutos_csv/
cp -a "/bolsa/validacion/"$(ls /bolsa/validacion/ | grep 'futuro1')"/." "/bolsa/validacion_datos/futuro1_brutos_csv/"
mkdir -p /bolsa/validacion_datos/futuro2_brutos_csv/
cp -a "/bolsa/validacion/"$(ls /bolsa/validacion/ | grep 'futuro2')"/." "/bolsa/validacion_datos/futuro2_brutos_csv/"


##########################################################################################################
#001 Ganancia leve en corto plazo
/home/carloslinux/Desktop/GIT_BOLSA/BolsaScripts/temp_validacionEuros_001.sh

#002 Ganancia grande en corto plazo
/home/carloslinux/Desktop/GIT_BOLSA/BolsaScripts/temp_validacionEuros_002.sh

#003 Ganancia grande en medio plazo
/home/carloslinux/Desktop/GIT_BOLSA/BolsaScripts/temp_validacionEuros_003.sh

#004 Ganancia leve pero mantenida/estable en corto plazo
/home/carloslinux/Desktop/GIT_BOLSA/BolsaScripts/temp_validacionEuros_004.sh

#005 Ganancia grande y mantenida en corto plazo --> LA IDEAL, pero apenas hay casos, claro.
/home/carloslinux/Desktop/GIT_BOLSA/BolsaScripts/temp_validacionEuros_005.sh


