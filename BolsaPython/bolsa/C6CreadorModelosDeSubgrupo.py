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
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.utils import resample

print("---- CAPA 6 - Crear almacenar y evaluar varios modelos (para cada subgrupo) -------")
print("Tipo de problema: CLASIFICACION BINOMIAL (target es boolean)")

print ("PARAMETROS: ")
dir_csvs_entrada = sys.argv[1]
dirModelos = sys.argv[2]
modoDebug = True  #En modo debug se pintan los dibujos. En otro caso, se evita calculo innecesario
print ("dir_csvs_entrada: %s" % dir_csvs_entrada)
print ("dirModelos: %s" % dirModelos)


################# FUNCIONES ########################################
def ejecutarModeloyGuardarlo(nombreModelo, modelo, pathModelo, ds_train_f, ds_train_t, modoDebug):

    print("** SVC (SVM para Clasificacion) **")
    # URL: https://scikit-learn.org/stable/modules/svm.html
    nombreModelo = "svc"
    pathModelo = dirModelos + str(id_subgrupo) + "_" + nombreModelo + ".modelo"
    modelo = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False,
                     tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                     decision_function_shape='ovr', break_ties=False, random_state=None)
    modelo.fit(ds_train_f, ds_train_t)  # ENTRENAMIENTO (TRAIN)
    s = pickle.dump(modelo, open(pathModelo, 'wb'))


def cargarModeloyUsarlo(dirModelos, pathModelo, ds_test_f, ds_test_t, modoDebug):

    modelo_loaded = pickle.load(open(pathModelo, 'rb'))
    ds_test_t_pred = modelo_loaded.predict(ds_test_f)  # PREDICCION de los targets de TEST (los compararemos con los que tenemos)
    area_bajo_roc = roc_auc_score(ds_test_t, ds_test_t_pred)

    print(nombreModelo + ".roc_auc_score = " + str(round(area_bajo_roc, 4)))
    average_precision = average_precision_score(ds_test_t, ds_test_t_pred)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))

    if modoDebug:
        print("Curva ROC...")
        # EVALUACION DE MODELOS - Curva ROC: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
        fpr_modelo, tpr_modelo, _ = roc_curve(ds_test_t, ds_test_t_pred)
        path_dibujo = dirModelos + str(id_subgrupo) + "_" + nombreModelo + "_roc.png"
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_modelo, tpr_modelo, label=nombreModelo)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(nombreModelo + ' - Curva ROC', fontsize=10)
        plt.legend(loc='best')
        plt.savefig(path_dibujo, bbox_inches='tight')
        # Limpiando dibujo:
        plt.clf()
        plt.cla()
        plt.close()

        print("Matriz de confusion...")
        path_dibujo = dirModelos + str(id_subgrupo) + "_" + nombreModelo + "_matriz_conf.png"
        disp = plot_confusion_matrix(modelo_loaded, ds_test_f, ds_test_t, cmap=plt.cm.Blues, normalize=None)
        disp.ax_.set_title(nombreModelo)
        print(disp.confusion_matrix)
        plt.savefig(path_dibujo, bbox_inches='tight')
        # Limpiando dibujo:
        plt.clf()
        plt.cla()
        plt.close()

    return area_bajo_roc



################# MAIN ########################################
# GANADOR DEL SUBGRUPO (acumuladores)
ganador_nombreModelo = "NINGUNO"
ganador_area_bajo_roc = 0


print("Recorremos los CSVs (subgrupos) que hay en el DIRECTORIO...")
for entry in os.listdir(dir_csvs_entrada):
  print("entry: " + entry)
  path_absoluto_fichero = os.path.join(dir_csvs_entrada, entry)
  print("path_absoluto_fichero: " + path_absoluto_fichero)

  if (entry.endswith('.csv') and os.path.isfile(path_absoluto_fichero) and os.stat(path_absoluto_fichero).st_size > 0 ):
    print("-------------------------------------------------------------------------------------------------")
    id_subgrupo = Path(entry).stem
    print("id_subgrupo=" + id_subgrupo)
    pathEntrada = os.path.abspath(entry)
    print("Cargar datos (CSV reducido) de fichero: " + path_absoluto_fichero)
    inputFeaturesyTarget = pd.read_csv(path_absoluto_fichero, sep='|')
    print("inputFeaturesyTarget: " + str(inputFeaturesyTarget.shape[0]) + " x " + str(inputFeaturesyTarget.shape[1]))

    print("BALANCEAR los casos positivos y negativos, haciendo downsampling de la clase mayoritaria...")
    print("URL: https://elitedatascience.com/imbalanced-classes")
    ift_mayoritaria = inputFeaturesyTarget[inputFeaturesyTarget.TARGET == False] #En este caso los mayoritarios son los False
    ift_minoritaria = inputFeaturesyTarget[inputFeaturesyTarget.TARGET == True]
    print("ift_mayoritaria:" + str(ift_mayoritaria.shape[0]) + " x " + str(ift_mayoritaria.shape[1]))
    print("ift_minoritaria:" + str(ift_minoritaria.shape[0]) + " x " + str(ift_minoritaria.shape[1]))
    num_muestras_minoria = ift_minoritaria.shape[0]
    print("num_muestras_minoria: " + str(num_muestras_minoria))
    ift_mayoritaria_downsampled = resample(ift_mayoritaria, replace=False, n_samples=num_muestras_minoria, random_state=123)

    ift_balanceadas = pd.concat([ift_mayoritaria_downsampled, ift_minoritaria]) # Juntar ambas clases ya BALANCEADAS
    print("Las clases ya están balanceadas:")
    ift_balanceadas.TARGET.value_counts()


    print("DIVIDIR EL DATASET DE ENTRADA EN 3 PARTES: TRAIN (60%), TEST (20%), VALIDACION (20%)...")
    ds_train, ds_test, ds_validacion = np.split(ift_balanceadas.sample(frac=1), [int(.6 * len(ift_balanceadas)), int(.8 * len(ift_balanceadas))])
    print("TRAIN --> " + str(ds_train.shape[0]) + " x " + str(ds_train.shape[1]))
    print("TEST --> " + str(ds_test.shape[0]) + " x " + str(ds_test.shape[1]))
    print("VALIDACION --> " + str(ds_validacion.shape[0]) + " x " + str(ds_validacion.shape[1]))

    print("Separamos FEATURES y TARGETS, de los 3 dataframes...")
    ds_train_f = ds_train.drop('TARGET', axis=1).to_numpy()
    ds_train_t = ds_train[['TARGET']].to_numpy()
    ds_test_f = ds_test.drop('TARGET', axis=1).to_numpy()
    ds_test_t = ds_test[['TARGET']].to_numpy()
    ds_validac_f = ds_validacion.drop('TARGET', axis=1).to_numpy()
    ds_validac_t = ds_validacion[['TARGET']].to_numpy()

    print("---------------- MODELOS con varias configuraciones (hiperparametros) --------")

    print("MODELOS: " + "https://scikit-learn.org/stable/supervised_learning.html")
    print("EVALUACION de los modelos con: " + "https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics")
    print("EVALUACION con curva precision-recall: " + "https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html")


    nombreModelo = "svc"
    pathModelo = dirModelos + str(id_subgrupo) + "_" + nombreModelo + ".modelo"
    modelo = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False,
                     tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                     decision_function_shape='ovr', break_ties=False, random_state=None)
    ejecutarModeloyGuardarlo(nombreModelo, modelo, pathModelo, ds_train_f, ds_train_t, modoDebug)
    area_bajo_roc = cargarModeloyUsarlo(dirModelos, pathModelo, ds_test_f, ds_test_t, modoDebug)
    print(type(area_bajo_roc))
    if area_bajo_roc > ganador_area_bajo_roc:
        ganador_area_bajo_roc = area_bajo_roc
        ganador_nombreModelo = nombreModelo


    nombreModelo = "logreg"
    pathModelo = dirModelos + str(id_subgrupo) + "_" + nombreModelo + ".modelo"
    modelo = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr')
    ejecutarModeloyGuardarlo(nombreModelo, modelo, pathModelo, ds_train_f, ds_train_t, modoDebug)
    area_bajo_roc = cargarModeloyUsarlo(dirModelos, pathModelo, ds_test_f, ds_test_t, modoDebug)
    print(type(area_bajo_roc))
    if area_bajo_roc > ganador_area_bajo_roc:
        ganador_area_bajo_roc = area_bajo_roc
        ganador_nombreModelo = nombreModelo

    nombreModelo = "rf"
    pathModelo = dirModelos + str(id_subgrupo) + "_" + nombreModelo + ".modelo"
    modelo = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    ejecutarModeloyGuardarlo(nombreModelo, modelo, pathModelo, ds_train_f, ds_train_t, modoDebug)
    area_bajo_roc = cargarModeloyUsarlo(dirModelos, pathModelo, ds_test_f, ds_test_t, modoDebug)
    print(type(area_bajo_roc))
    if area_bajo_roc > ganador_area_bajo_roc:
        ganador_area_bajo_roc = area_bajo_roc
        ganador_nombreModelo = nombreModelo

    nombreModelo = "nn"
    pathModelo = dirModelos + str(id_subgrupo) + "_" + nombreModelo + ".modelo"
    modelo = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    ejecutarModeloyGuardarlo(nombreModelo, modelo, pathModelo, ds_train_f, ds_train_t, modoDebug)
    area_bajo_roc = cargarModeloyUsarlo(dirModelos, pathModelo, ds_test_f, ds_test_t, modoDebug)
    print(type(area_bajo_roc))
    if area_bajo_roc > ganador_area_bajo_roc:
        ganador_area_bajo_roc = area_bajo_roc
        ganador_nombreModelo = nombreModelo


  print("********* GANADOR *************")
  print("El modelo ganador es " + ganador_nombreModelo +" con un area_bajo_ROC de " + str(round(ganador_area_bajo_roc, 4)))


############################################################
print("------------ FIN ----------------")


