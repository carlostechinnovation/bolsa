#!/bin/bash

echo -e "INVERSION POSTERIORI - INICIO: "$( date "+%Y%m%d%H%M%S" )



1º. inversion.sh ha generado unos 100 ficheros al dia: 50 son de 300KB (tienen un pequeño historico) y otros son de 5KB (antiguedad=0)

2º. Entonces, en PYTHON, vamos a recorrer todos los CSV de 5KB y vamos a construir una tabla (DATAFRAME) que tenga -->  DIA_INVERSION, DIA_POSTERIORI, subgrupo, empresa, target_predicho, target_predicho_PROB

3º. En PYTHON, vamos a recorrer todos los CSVs de 300 KB y vamos a construir una tabla (DATAFRAME) que tenga --> DIA, EMPRESA, precio_close, target (es el target REAL)

4º. Comparamos las dos tablas --> En la tabla 1 metemos la columna de qué ha pasado realmente en el futuro (precio, target REAL) 

5º. Graficas del rendimiento...


echo -e "INVERSION POSTERIORI - FIN: "$( date "+%Y%m%d%H%M%S" )








