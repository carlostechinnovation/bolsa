#!/bin/bash

function buscarEmpresaEnSubgrupos () {
	dirSubgrupos="${1}"
	empresa="${2}"
	ficheroLog="${3}"
	cadena=""
	for nombreSubg in $(ls ${dirSubgrupos})
	do
		pathEmpresasDeSg="/bolsa/pasado/subgrupos/${nombreSubg}/EMPRESAS.txt"
		echo "Analizando si la empresa ${empresa} esta en el fichero ${pathEmpresasDeSg} ..."
		pattern="_${empresa}\."
		encontrado=$(cat ${pathEmpresasDeSg} | grep "$pattern")
		if [ -z "${encontrado}" ];then
			echo "$encontrado is empty"
		else
			echo "$encontrado is NOT empty"
			cadena="${cadena} ${nombreSubg}"
		fi
	done
	
	echo "La empresa aparece en estos subgrupos: ${cadena}"
}


buscarEmpresaEnSubgrupos "/bolsa/pasado/subgrupos/" "AMRS" "./prueba.txt"






