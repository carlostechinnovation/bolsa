from os import sep
from pathlib import Path
from os.path import exists
import sqlite3
import pandas as pd
from tabulate import tabulate
import os

# UTILIDADES:
# https://mungingdata.com/sqlite/create-database-load-csv-python/

LOG_BASEDATOS="/bolsa/logs/basedatos.log"
DIR_PROGRAMAS="/home/carloslinux/Desktop/PROGRAMAS/"
PATH_BASEDATOS="/bolsa/basedatos/bolsa.db"

################################### FUNCIONES ###################################
def importarCsvHaciaTabla(conn, pathCSV, pathBaseDeDatosYaAbierta, tabla):
    print("Importando...   CSV=", pathCSV," --> TABLA=", tabla)
    df = pd.read_csv(pathCSV, sep='|', engine='python')
    df.to_sql(tabla, conn, if_exists='replace', index=False)
    consulta="SELECT * FROM " + tabla + " LIMIT 1"
    fila = c.execute(consulta)
    # print(tabulate(fila, headers='keys', tablefmt='psql'))


######################################################################
# INSTALACION
#sudo apt install sqlite3

######################################################################
print("Creación de base de datos (es un fichero): ", PATH_BASEDATOS)
baseDeDatosExiste = exists(PATH_BASEDATOS)
if baseDeDatosExiste == False:
    print("No existe. La generamos...")
    Path(PATH_BASEDATOS).touch()  # la creamos
else:
    print("Existe previamente. La borramos para regenerarla...")
    os.remove(PATH_BASEDATOS)
    Path(PATH_BASEDATOS).touch()  # la creamos


print("Conexion a base de datos...")
with sqlite3.connect(PATH_BASEDATOS) as conn:
    print("Cursor...")
    c = conn.cursor()

    #-------------------------------------------------
    # print("Se crea tabla de pruebas...")
    # salida_pruebas1 = c.execute('''DROP TABLE IF EXISTS pruebas;''')
    # salida_pruebas2 = c.execute('''CREATE TABLE pruebas (id INTEGER PRIMARY KEY, nombre TEXT);''')
    # salida_pruebas3 = c.execute('''INSERT INTO pruebas (id,nombre) VALUES (1, 'empresa1');''')
    # salida_pruebas4 = c.execute('''INSERT INTO pruebas (id,nombre) VALUES (2, 'empresa2');''')
    # salida_pruebas5 = c.execute('''SELECT * FROM pruebas LIMIT 10;''')
    # df = pd.read_sql_query('''SELECT * FROM pruebas LIMIT 10;''', conn)
    #print(tabulate(df, headers='keys', tablefmt='psql'))

    #-------------------------------------------------
    print("############# CARGAR CSV en la BASE DE DATOS ####################")

    print("PASADO - elaborados:")
    with os.scandir("/bolsa/pasado/elaborados/") as iterador:
        for PATH_CSV in iterador:
            if PATH_CSV.name.endswith(".csv") and "/NASDAQ_" in PATH_CSV.path and PATH_CSV.is_file():
                importarCsvHaciaTabla(conn, PATH_CSV.path, PATH_BASEDATOS, "pasado_elaborados_" + PATH_CSV.name.replace(".csv", "").replace("NASDAQ_", ""))

    print("FUTURO - elaborados:")
    with os.scandir("/bolsa/futuro/elaborados/") as iterador:
        for PATH_CSV in iterador:
            if PATH_CSV.name.endswith(".csv") and "/NASDAQ_" in PATH_CSV.path and PATH_CSV.is_file():
                importarCsvHaciaTabla(conn, PATH_CSV.path, PATH_BASEDATOS,
                                      "futuro_elaborados_" + PATH_CSV.name.replace(".csv", "").replace("NASDAQ_", ""))

    print("PASADO - SUBGRUPOS:")
    for root, dirs, files in os.walk("/bolsa/pasado/subgrupos/"):
        for filename in files:
            if filename.__eq__('REDUCIDO.csv'):
                pathcompleto = root + "/" + filename
                nombreSubgrupo = pathcompleto.rsplit('/', 2)[1]
                importarCsvHaciaTabla(conn, pathcompleto, PATH_BASEDATOS,
                                      "pasado_" + nombreSubgrupo + "_" + filename.replace(".csv", "").lower())

    print("FUTURO - SUBGRUPOS:")
    for root, dirs, files in os.walk("/bolsa/futuro/subgrupos/"):
        for filename in files:
            if filename.__eq__('REDUCIDO.csv'):
                pathcompleto = root + "/" + filename
                nombreSubgrupo = pathcompleto.rsplit('/', 2)[1]
                importarCsvHaciaTabla(conn, pathcompleto, PATH_BASEDATOS,
                                      "pasado_" + nombreSubgrupo + "_" + filename.replace(".csv", "").lower())
    #-------------------------------------------------


    print("Cerrando conexiones...")
    c.close()
    conn.commit()
    #conn.close()

print("FIN")
