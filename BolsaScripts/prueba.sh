#!/bin/bash
ACTIVAR_DESCARGA="${1}" #S o N
if [ "${ACTIVAR_DESCARGA}" = "S" ];  then
	echo "dentro"
fi


tamanioorigen=$(du -s "/bolsa/validacion/pasado_brutos_csv/" | cut -f1)
echo "${tamanioorigen}"
if [ ${tamanioorigen} > 0 ]; then
	echo "Hay datos"
fi

