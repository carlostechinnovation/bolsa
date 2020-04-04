import pandas as pd
import matplotlib.pyplot as plt

print("Pruebas-Inicio")


df = pd.read_csv('/home/carloslinux/Desktop/GIT_BOLSA/BolsaPython/ejemplos/pruebaDibujoDatos.csv')
ax = plt.gca()  # gca significa 'get current axis'
df.plot(kind='line',x='aniomesdia',y='SG_1_precision',ax=ax, legend=True, marker="+")
df.plot(kind='line',x='aniomesdia',y='SG_2_precision', color='red', ax=ax, legend=True, marker="+")
ax.tick_params(axis='x', labelrotation=20)  # Rota las etiquetas del eje X
ax.ticklabel_format(useOffset=False)  # Evita notacion cientifica
plt.title('Evolución de la PRECISION cada modelo entrenado de subgrupo')
#plt.show()
plt.savefig('/home/carloslinux/Desktop/GIT_BOLSA/BolsaPython/ejemplos/pruebaDibujoSalida.png')


print("Pruebas-Fin")
