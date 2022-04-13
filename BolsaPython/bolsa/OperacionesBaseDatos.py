from os import sep
from pathlib import Path
from os.path import exists
import sqlite3
import pandas as pd
from tabulate import tabulate

# UTILIDADES:
# https://mungingdata.com/sqlite/create-database-load-csv-python/

LOG_BASEDATOS="/bolsa/logs/basedatos.log"
DIR_PROGRAMAS="/home/carloslinux/Desktop/PROGRAMAS/"
PATH_BASEDATOS="/bolsa/basedatos/bolsa.db"

################################### FUNCIONES ###################################
def importarCsvHaciaTabla(pathCSV, basededatos, tabla):
    print("importarCsvHaciaTabla: ", pathCSV,"->", basededatos,".", tabla)


######################################################################
# INSTALACION
#sudo apt install sqlite3

######################################################################

print("Creación de base de datos (es un fichero): ", PATH_BASEDATOS)
baseDeDatosExiste = exists(PATH_BASEDATOS)
if baseDeDatosExiste == False:
    Path(PATH_BASEDATOS).touch()  # la creamos

print("Conexion a base de datos...")
with sqlite3.connect(PATH_BASEDATOS) as conn:
    print("Cursor...")
    c=conn.cursor()

    #-------------------------------------------------
    print("Se crea tabla de pruebas...")
    salida_pruebas1 = c.execute('''DROP TABLE IF EXISTS pruebas;''')
    salida_pruebas2 = c.execute('''CREATE TABLE pruebas (id INTEGER PRIMARY KEY, nombre TEXT);''')
    salida_pruebas3 = c.execute('''INSERT INTO pruebas (id,nombre) VALUES (1, 'empresa1');''')
    salida_pruebas4 = c.execute('''INSERT INTO pruebas (id,nombre) VALUES (2, 'empresa2');''')
    salida_pruebas5 = c.execute('''SELECT * FROM pruebas LIMIT 10;''')
    df=pd.read_sql_query('''SELECT * FROM pruebas LIMIT 10;''', conn)
    print(tabulate(df, headers='keys', tablefmt='psql'))

    #-------------------------------------------------
    print("############# CARGAR CSV en la BASE DE DATOS ####################")
    PATH_CSV ="/bolsa/pasado/limpios/NASDAQ_AAPL.csv"
    df = pd.read_csv(PATH_CSV, sep='|', engine='python')
    print("Write the data to a sqlite table")
    df.to_sql('pasado_limpios_NASDAQ_AAPL', conn, if_exists='replace', index=False)
    #-------------------------------------------------

    print("Create a cursor object")
    print("Fetch and display result")
    for row in c.execute('SELECT * FROM pasado_limpios_NASDAQ_AAPL'):
        print(row)
        break

    print("Cerrando conexiones...")
    c.close()
    conn.commit()
    conn.close()

print("FIN")
