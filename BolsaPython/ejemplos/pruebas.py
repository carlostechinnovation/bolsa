import sys
import os
import pandas as pd

print("--- PRUEBAS ---")
mis_datos = pd.read_csv(filepath_or_buffer="/bolsa/pasado/subgrupos/SG_11/COMPLETO.csv", sep='|')
print(mis_datos.head().to_string())
print(mis_datos.describe().to_string())


print("--- FINAL ---")