#!/bin/bash

LOG_INTEGRACION="/bolsa/logs/integracion.log"
echo -e ""  > ${LOG_INTEGRACION}

echo -e "Damos por hecho que tenemos una ejecucion completa de validador (1 pasado y 2 futuros)."  >> ${LOG_INTEGRACION}
echo -e "Cogemos una empresa y vemos su evolucion en cada capa."  >> ${LOG_INTEGRACION}


echo -e "******************************** Test del PASADO *******************************" >> ${LOG_INTEGRACION}
empresa="ADXS"

echo -e "\n-- Parametros de escenario ---" >> ${LOG_INTEGRACION}
LOG_PASADO=$(ls /bolsa/logs/ | grep 'pasado')
echo -e "LOG_PASADO: ${LOG_PASADO}"  >> ${LOG_INTEGRACION}
cat "/bolsa/logs/${LOG_PASADO}" | grep 'PARAMETROS_MASTER' >> ${LOG_INTEGRACION}

echo -e "\n-- capa 1 (brutos) ---" >> ${LOG_INTEGRACION}
BRUTO_YF="/bolsa/pasado/brutos/YF_NASDAQ_${empresa}.txt"
BRUTO_FZ="/bolsa/pasado/brutos/FZ_NASDAQ_${empresa}.html"
echo -e "Bruto - YahooFinance: ${BRUTO_YF} --> Tamanio (bytes) = "$(stat -c%s "$BRUTO_YF") >> ${LOG_INTEGRACION}
echo -e "Bruto - Finviz: ${BRUTO_FZ} --> Tamanio (bytes) = "$(stat -c%s "$BRUTO_FZ") >> ${LOG_INTEGRACION}
BRUTO_CSV="/bolsa/pasado/brutos_csv/NASDAQ_${empresa}.csv"
echo -e "Bruto CSV: ${BRUTO_CSV} --> Tamanio (bytes) = "$(stat -c%s "$BRUTO_CSV")" con "$(wc -l $BRUTO_CSV | cut -d\  -f 1)" filas\n" >> ${LOG_INTEGRACION}
head -n 5 ${BRUTO_CSV}>> ${LOG_INTEGRACION}

echo -e "\n-- capa 2 (limpios) ---" >> ${LOG_INTEGRACION}
LIMPIO="/bolsa/pasado/limpios/NASDAQ_${empresa}.csv"
echo -e "Limpio: ${LIMPIO} --> Tamanio (bytes) = "$(stat -c%s "$LIMPIO")" con "$(wc -l $LIMPIO | cut -d\  -f 1)" filas\n" >> ${LOG_INTEGRACION}
head -n 5 ${LIMPIO}>> ${LOG_INTEGRACION}

echo -e "\n-- capa 3 (elaboradas) ---" >> ${LOG_INTEGRACION}
ELABORADO="/bolsa/pasado/elaborados/NASDAQ_${empresa}.csv"
echo -e "Elaborado: ${ELABORADO} --> Tamanio (bytes) = "$(stat -c%s "$ELABORADO")" con "$(wc -l $ELABORADO | cut -d\  -f 1)" filas\n" >> ${LOG_INTEGRACION}
head -n 5 ${ELABORADO}>> ${LOG_INTEGRACION}

echo -e "\n-- capa 4 (subgrupos) ---" >> ${LOG_INTEGRACION}
DIR_SUBGRUPOS="/bolsa/pasado/subgrupos/"
echo -e "Subgrupos creados (los que superan suficientes requisitos) -> "$(ls $DIR_SUBGRUPOS)"\n" >> ${LOG_INTEGRACION}

SG_ANALIZADO="SG_11"

SG_ENTRADA="${DIR_SUBGRUPOS}${SG_ANALIZADO}/COMPLETO.csv"
echo -e "Subgrupo ${SG_ANALIZADO} - Datos entrada: ${SG_ENTRADA} --> Tamanio (bytes) = "$(stat -c%s "$SG_ENTRADA")" con "$(wc -l $SG_ENTRADA | cut -d\  -f 1)" filas\n" >> ${LOG_INTEGRACION}
head -n 5 ${SG_ENTRADA}>> ${LOG_INTEGRACION}

SG_REDUCIDO="${DIR_SUBGRUPOS}${SG_ANALIZADO}/REDUCIDO.csv"
echo -e "\nSubgrupo ${SG_ANALIZADO} - Datos reducidos (normalizar + seleccion de columnas): ${SG_REDUCIDO} --> Tamanio (bytes) = "$(stat -c%s "$SG_REDUCIDO")" con "$(wc -l $SG_REDUCIDO | cut -d\  -f 1)" filas\n" >> ${LOG_INTEGRACION}
head -n 5 ${SG_REDUCIDO}>> ${LOG_INTEGRACION}

echo -e "\nSubgrupo ${SG_ANALIZADO} - Modelo ganador --> "$(ls ${DIR_SUBGRUPOS}${SG_ANALIZADO}/ | grep 'ganador')"\n" >> ${LOG_INTEGRACION}


echo -e "\n******** FIN del test de integracion **************" >> ${LOG_INTEGRACION}


