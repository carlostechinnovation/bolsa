import os.path
import subprocess

print("BUCLE - INICIO")
##################################################################################################
print("PARAMETROS: ")
dir_subgrupo_prefijo = "/bolsa/pasado/subgrupos/"
modoTiempo = "pasado"
maxFeatReducidas = "35"
maxFilasEntrada = "25000"
desplazamientoAntiguedad = "0"
##################################################################################################
for numero in range(1, 60, 1):
    dir_subgrupo = dir_subgrupo_prefijo + "SG_" + str(numero) + "/"
    comando = os.getcwd() + "/C5C6Manual.py " + dir_subgrupo + " " + modoTiempo + " " + maxFeatReducidas + " " + maxFilasEntrada + " " + desplazamientoAntiguedad
    if os.path.exists(dir_subgrupo):
        print("comando " + comando)
        subprocess.call("python " + comando, shell=True)

        print("BUCLE - FIN")
