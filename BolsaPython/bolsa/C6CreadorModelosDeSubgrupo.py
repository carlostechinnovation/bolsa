import sys
import os
import pandas as pd
import numpy as np
np.random.seed(10)
from pathlib import Path
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

print("---- CAPA 6 - Crear almacenar y evaluar varios modelos (para cada subgrupo) -------")
print("Tipo de problema: CLASIFICACION BINOMIAL (target es boolean)")

print ("PARAMETROS: ")
dir_csvs_entrada = sys.argv[1]
pathModelos = sys.argv[2]
print ("dir_csvs_entrada: %s" % dir_csvs_entrada)
print ("pathModelos: %s" % pathModelos)


print("Recorremos los CSVs que hay en el DIRECTORIO...")
for entry in os.listdir(dir_csvs_entrada):
  print("entry: " + entry)
  path_absoluto_fichero = os.path.join(dir_csvs_entrada, entry)
  print("path_absoluto_fichero: " + path_absoluto_fichero)

  if (entry.endswith('.csv') and os.path.isfile(path_absoluto_fichero) and os.stat(path_absoluto_fichero).st_size > 0 ):
    print("-------------------------------------------------------------------------------------------------")
    id_subgrupo = Path(entry).stem
    print("id_subgrupo=" + id_subgrupo)
    pathEntrada = os.path.abspath(entry)
    print("Cargar datos (CSV reducido) de fichero: " + pathEntrada)
    inputFeaturesyTarget = pd.read_csv(path_absoluto_fichero, sep='|')

    print("DIVIDIR EL DATASET DE ENTRADA EN 3 PARTES: TRAIN (60%), TEST (20%), VALIDACION (20%)...")
    ds_train, ds_test, ds_validacion = np.split(inputFeaturesyTarget.sample(frac=1), [int(.6 * len(inputFeaturesyTarget)), int(.8 * len(inputFeaturesyTarget))])
    print("TRAIN --> " + str(ds_train.shape[0]) + " x " + str(ds_train.shape[1]))
    print("TEST --> " + str(ds_test.shape[0]) + " x " + str(ds_test.shape[1]))
    print("VALIDACION --> " + str(ds_validacion.shape[0]) + " x " + str(ds_validacion.shape[1]))

    print("Separamos FEATURES y TARGETS, de los 3 datasets...")
    ds_train_f = ds_train.drop('TARGET', axis=1)
    ds_train_t = ds_train[['TARGET']]
    ds_test_f = ds_test.drop('TARGET', axis=1)
    ds_test_t = ds_test[['TARGET']]
    ds_validac_f = ds_validacion.drop('TARGET', axis=1)
    ds_validac_t = ds_validacion[['TARGET']]

    print("Crear CARPETA para guardar los MODELOS...")
    dir_modelos_subgrupo = pathModelos + str(id_subgrupo)
    print("dir_modelos_subgrupo: " + dir_modelos_subgrupo+"/")
    if os.path.exists(dir_modelos_subgrupo):  # Borrar si existe
        os.rmdir(dir_modelos_subgrupo)
    if not os.path.exists(dir_modelos_subgrupo):  # Crear carpeta
        os.mkdir(dir_modelos_subgrupo)

    print("---------------- MODELOS con varias configuraciones (hiperparametros) --------")

    print("** SVC (SVM para Clasificacion) **")
    # URL: https://scikit-learn.org/stable/modules/svm.html
    nombreModelo="svc"
    pathModelo = dir_modelos_subgrupo + "_svc.modelo"
    modelo = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False,
                     tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                     decision_function_shape='ovr', break_ties=False, random_state=None)
    modelo.fit(ds_train_f, ds_train_t)
    s = pickle.dump(modelo, open(pathModelo, 'wb'))
    modelo_loaded = pickle.load(open(pathModelo, 'rb'))
    ds_test_t_pred = modelo_loaded.predict(ds_test_f)
    print(nombreModelo + ".score = " + str(round(modelo_loaded.score(ds_test_f, ds_test_t_pred), 4)))
    print(nombreModelo + ".roc_auc_score = " + str(round(roc_auc_score(ds_test_t, ds_test_t_pred), 4)))
    fpr_modelo, tpr_modelo, _ = roc_curve(ds_test_t_pred, ds_test_t_pred)

    # EVALUACION DE MODELOS - Curva ROC: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_modelo, tpr_modelo, label=nombreModelo)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()


    print("** REGRESION LOGISTICA (para Clasificacion) **")
    # URL:
    nombreModelo = "logreg"
    pathModelo = dir_modelos_subgrupo + "_logreg.modelo"
    modelo = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr')
    modelo.fit(ds_train_f, ds_train_t)
    s = pickle.dump(modelo, open(pathModelo, 'wb'))
    modelo_loaded = pickle.load(open(pathModelo, 'rb'))
    ds_test_t_pred = modelo_loaded.predict(ds_test_f)
    print(nombreModelo + ".score = " + str(round(modelo_loaded.score(ds_test_f, ds_test_t_pred), 4)))
    print(nombreModelo + ".roc_auc_score = " + str(round(roc_auc_score(ds_test_t, ds_test_t_pred), 4)))

    print("** RANDOM FOREST (para Clasificacion) **")
    # URL:
    nombreModelo = "rf"
    pathModelo = dir_modelos_subgrupo + "_rf.modelo"
    modelo = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    modelo.fit(ds_train_f, ds_train_t)
    s = pickle.dump(modelo, open(pathModelo, 'wb'))
    modelo_loaded = pickle.load(open(pathModelo, 'rb'))
    ds_test_t_pred = modelo_loaded.predict(ds_test_f)
    print(nombreModelo + ".score = " + str(round(modelo_loaded.score(ds_test_f, ds_test_t_pred), 4)))
    print(nombreModelo + ".roc_auc_score = " + str(round(roc_auc_score(ds_test_t, ds_test_t_pred), 4)))

    print("** RED NEURONAL (para Clasificacion) **")
    # URL:
    nombreModelo = "nn"
    pathModelo = dir_modelos_subgrupo + "_nn.modelo"
    modelo = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    modelo.fit(ds_train_f, ds_train_t)
    s = pickle.dump(modelo, open(pathModelo, 'wb'))
    modelo_loaded = pickle.load(open(pathModelo, 'rb'))
    ds_test_t_pred = modelo_loaded.predict(ds_test_f)
    print(nombreModelo + ".score = " + str(round(modelo_loaded.score(ds_test_f, ds_test_t_pred), 4)))
    print(nombreModelo + ".roc_auc_score = " + str(round(roc_auc_score(ds_test_t, ds_test_t_pred), 4)))


################## MAIN ########################################
print("PENDIENTE...")



############################################################
print("------------ FIN ----------------")


