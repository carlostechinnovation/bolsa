import sys
import os
import pandas as pd

print("--- PRUEBAS ---")
entradaFeaturesYTarget = pd.read_csv(filepath_or_buffer="/bolsa/futuro/subgrupos/SG_6/COMPLETO.csv", sep='|')
print(entradaFeaturesYTarget)
ift_minoritaria = entradaFeaturesYTarget[entradaFeaturesYTarget.TARGET == True]
print("ift_minoritaria:" + str(ift_minoritaria.shape[0]) + " x " + str(ift_minoritaria.shape[1]))
num_copias_anhadidas=2
ift_minoritaria2 = ift_minoritaria.append([ift_minoritaria]*num_copias_anhadidas, ignore_index=True)
print("ift_minoritaria2:" + str(ift_minoritaria2.shape[0]) + " x " + str(ift_minoritaria2.shape[1]))


