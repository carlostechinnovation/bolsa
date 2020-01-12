#!/bin/bash

LOG_INTEGRACION="/bolsa/logs/integracion.log"
echo -e ""  > ${LOG_INTEGRACION}

echo -e "Damos por hecho que tenemos una ejecucion completa de validador (1 pasado y 2 futuros)."  >> ${LOG_INTEGRACION}
echo -e "Cogemos una empresa y vemos su evolucion en cada capa."  >> ${LOG_INTEGRACION}

echo -e "******** Test del PASADO **************" >> ${LOG_INTEGRACION}
empresa="ADXS"

echo -e "\n-- capa 1 (brutos) ---" >> ${LOG_INTEGRACION}
BRUTO_YF="/bolsa/pasado/brutos/YF_NASDAQ_${empresa}.txt"
BRUTO_FZ="/bolsa/pasado/brutos/FZ_NASDAQ_${empresa}.html"
echo -e "Bruto - YahooFinance: ${BRUTO_YF} --> Tamanio (bytes) = "$(stat -c%s "$BRUTO_YF") >> ${LOG_INTEGRACION}
echo -e "Bruto - Finviz: ${BRUTO_FZ} --> Tamanio (bytes) = "$(stat -c%s "$BRUTO_FZ") >> ${LOG_INTEGRACION}
BRUTO_CSV="/bolsa/pasado/brutos_csv/NASDAQ_${empresa}.csv"
echo -e "Bruto CSV: ${BRUTO_CSV} --> Tamanio (bytes) = "$(stat -c%s "$BRUTO_CSV")"\n" >> ${LOG_INTEGRACION}
head -n 5 ${BRUTO_CSV}>> ${LOG_INTEGRACION}

echo -e "\n-- capa 2 (limpios) ---" >> ${LOG_INTEGRACION}
LIMPIO="/bolsa/pasado/limpios/NASDAQ_${empresa}.csv"
echo -e "Limpio: ${LIMPIO} --> Tamanio (bytes) = "$(stat -c%s "$LIMPIO")"\n" >> ${LOG_INTEGRACION}
head -n 5 ${LIMPIO}>> ${LOG_INTEGRACION}

echo -e "\n-- capa 3 (elaboradas) ---" >> ${LOG_INTEGRACION}
LIMPIO="/bolsa/pasado/elaborados/NASDAQ_${empresa}.csv"
echo -e "Limpio: ${LIMPIO} --> Tamanio (bytes) = "$(stat -c%s "$LIMPIO")"\n" >> ${LOG_INTEGRACION}
head -n 5 ${LIMPIO}>> ${LOG_INTEGRACION}


echo -e "******** FIN del test de integracion **************" >> ${LOG_INTEGRACION}


