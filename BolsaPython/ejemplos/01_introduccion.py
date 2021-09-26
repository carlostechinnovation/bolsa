from sklearn import datasets

print("----------- DATOS (pasado + futuro) ----------")
print("Datasets de ejemplo")
iris = datasets.load_iris()
digits = datasets.load_digits()
print("digits.data:")
print(digits.data)
print("digits.target:")
print(digits.target)

print("----------- PASADO ----------")
from sklearn import svm
print("Creamos un modelo matematico SVM (vector machine) con unos hyperparametros concretos...")
pathModelo = "/bolsa/modelos/subgrupoEJEMPLO/svc_001"
modelo001 = svm.SVC(gamma=0.001, C=100.)
print("Entrenamiento (fit) con FEATURES (me dejo la última fila, para el ejemplo) + sus TARGETS...")
features_pasado = digits.data[:-1]
targets_pasado = digits.target[:-1]
modelo001.fit(features_pasado, targets_pasado)
print("Guardamos a disco el modelo ENTRENADO...")
from joblib import dump, load
s = dump(modelo001, pathModelo)

print("----------- FUTURO ----------")
print("Predicción del futuro: cargamos el modelo ENTRENADO guardado...")
modeloLeido = load(pathModelo)
print("Predicción del target para la última fila (para el ejemplo)...")
features_futuro=digits.data[-1:]
targets_futuro=modeloLeido.predict(features_futuro)
print(*targets_futuro, sep = ", ")

