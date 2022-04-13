#!/bin/bash

PROGRAMAS_CARLOS="/home/carloslinux/Desktop/PROGRAMAS/"

##### PROGRAMA GESTOR BASE DE DATOS #############
cd ${PROGRAMAS_CARLOS}sqlite/
sqlite3 ".open ${PROGRAMAS_CARLOS}sqlite/db/bolsa.db"  # Abre la base de datos (schema). Si no existe, la crea.

#sqlite3 .help

################# IMPORTACION DE DATOS HACIA TABLAS ################
#/home/carloslinux/Desktop/GIT_BOLSA/bolsa/BolsaScripts/
sqlite3 -batch /bolsa/basedatos/bolsa.db -separator '|' -header "DROP TABLE IF EXISTS pasado_limpios_NASDAQ_AAPL;"

#IMPORTANTE- Son dos comandos a la vez:
sqlite3 -batch /bolsa/basedatos/bolsa.db  -separator '|' -cmd ".import --csv -v /bolsa/pasado/limpios/NASDAQ_AAPL.csv pasado_limpios_NASDAQ_AAPL" ".quit" 

sqlite3 -batch /bolsa/basedatos/bolsa.db -header "SELECT 'pasado_limpios_NASDAQ_AAPL' AS tabla, count(*) AS numero_filas FROM pasado_limpios_NASDAQ_AAPL LIMIT 3;"
sqlite3 -batch /bolsa/basedatos/bolsa.db -header "SELECT * FROM pasado_limpios_NASDAQ_AAPL LIMIT 3;"

######## Visualizar la base de datos (es un fichero físico sencillo) #############
ls -lrt /bolsa/basedatos/bolsa.db

############################### CLIENTE ##############################
#Programa para visualizar fácilmente las tablas
${PROGRAMAS_CARLOS}SQLiteStudio/sqlitestudio &



