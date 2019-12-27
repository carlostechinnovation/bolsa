import sys
import os
import pandas as pd
import numpy as np
np.random.seed(12345)
from pathlib import Path
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

print("-------------------------------------------------------------------------------------------------")
print("---- CAPA 6 - Crear almacenar y evaluar varios modelos (para cada subgrupo) -------")
print("Tipo de problema: CLASIFICACION BINOMIAL (target es boolean)")

print ("PARAMETROS: ")
entrada_csv_subgrupo = sys.argv[1]
dirModelos = sys.argv[2]
modoDebug = True  #En modo debug se pintan los dibujos. En otro caso, se evita calculo innecesario
umbralCasosSuficientesClasePositiva = 100
print ("entrada_csv_subgrupo: %s" % entrada_csv_subgrupo)
print ("dirModelos: %s" % dirModelos)


################# FUNCIONES ########################################
def ejecutarModeloyGuardarlo(nombreModelo, modelo, pathModelo, ds_train_f, ds_train_t, modeloEsGrid, modoDebug):
    print("** " + nombreModelo + " **")
    out_grid_best_params = []
    modelo.fit(ds_train_f, ds_train_t)  # ENTRENAMIENTO (TRAIN)
    print("Se guarda el modelo " + nombreModelo + " en: " + pathModelo)
    if modeloEsGrid:
        s = pickle.dump(modelo.best_estimator_, open(pathModelo, 'wb'))
        out_grid_best_params = modelo.best_params_
        print("Modelo GRID tipo " + nombreModelo + " Los mejores parametros probados son: " + str(modelo.best_params_))
    else:
        s = pickle.dump(modelo, open(pathModelo, 'wb'))
    return out_grid_best_params


def cargarModeloyUsarlo(dirModelos, pathModelo, ds_test_f, ds_test_t, modoDebug):
    modelo_loaded = pickle.load(open(pathModelo, 'rb'))
    ds_test_t_pred = modelo_loaded.predict(ds_test_f)  # PREDICCION de los targets de TEST (los compararemos con los que tenemos)
    area_bajo_roc = roc_auc_score(ds_test_t, ds_test_t_pred)

    print(nombreModelo + ".roc_auc_score = " + str(round(area_bajo_roc, 4)))
    average_precision = average_precision_score(ds_test_t, ds_test_t_pred)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    recall=recall_score(ds_test_t, ds_test_t_pred, average='binary', pos_label=1)
    print('Average recall score: {0:0.2f}'.format(recall))

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
ganador_grid_mejores_parametros=[]


print("Recorremos los CSVs (subgrupos) que hay en el DIRECTORIO...")
entrada_csv_subgrupo

if (entrada_csv_subgrupo.endswith('.csv') and os.path.isfile(entrada_csv_subgrupo) and os.stat(entrada_csv_subgrupo).st_size > 0 ):

    id_subgrupo = Path(entrada_csv_subgrupo).stem
    print("id_subgrupo=" + id_subgrupo)
    pathEntrada = os.path.abspath(entrada_csv_subgrupo)
    print("Cargar datos (CSV reducido) de fichero: " + entrada_csv_subgrupo)
    inputFeaturesyTarget = pd.read_csv(entrada_csv_subgrupo, sep='|')
    print("inputFeaturesyTarget: " + str(inputFeaturesyTarget.shape[0]) + " x " + str(inputFeaturesyTarget.shape[1]))

    print("BALANCEAR los casos positivos y negativos, haciendo downsampling de la clase mayoritaria...")
    print("URL: https://elitedatascience.com/imbalanced-classes")
    ift_mayoritaria = inputFeaturesyTarget[inputFeaturesyTarget.TARGET == False] #En este caso los mayoritarios son los False
    ift_minoritaria = inputFeaturesyTarget[inputFeaturesyTarget.TARGET == True]
    print("ift_mayoritaria:" + str(ift_mayoritaria.shape[0]) + " x " + str(ift_mayoritaria.shape[1]))
    print("ift_minoritaria:" + str(ift_minoritaria.shape[0]) + " x " + str(ift_minoritaria.shape[1]))
    print("Tasa de desbalanceo entre clases = " + str(ift_mayoritaria.shape[0]) + "/" + str(ift_minoritaria.shape[0]) + " = " + str(ift_mayoritaria.shape[0]/ift_minoritaria.shape[0]))
    num_muestras_minoria = ift_minoritaria.shape[0]

    casosInsuficientes = (num_muestras_minoria < umbralCasosSuficientesClasePositiva)
    if(casosInsuficientes):
        print("Numero de casos en clase minoritaria es INSUFICIENTE. Así que abandonamos este dataset y seguimos")

    else:

        print("num_muestras_minoria: " + str(num_muestras_minoria))
        ift_mayoritaria_downsampled = resample(ift_mayoritaria, replace=False, n_samples=num_muestras_minoria, random_state=123)

        ift_balanceadas = pd.concat([ift_mayoritaria_downsampled, ift_minoritaria]) # Juntar ambas clases ya BALANCEADAS
        print("Las clases ya están balanceadas:")
        print("ift_balanceadas:" + str(ift_balanceadas.shape[0]) + " x " + str(ift_balanceadas.shape[1]))








        print("DIVIDIR EL DATASET DE ENTRADA EN 3 PARTES: TRAIN (70%), TEST (20%), VALIDACION (10%)...")
        ds_train, ds_test, ds_validacion = np.split(ift_balanceadas.sample(frac=1), [int(.7 * len(ift_balanceadas)), int(.9 * len(ift_balanceadas))])
        print("TRAIN --> " + str(ds_train.shape[0]) + " x " + str(ds_train.shape[1]))
        print("TEST --> " + str(ds_test.shape[0]) + " x " + str(ds_test.shape[1]))
        print("VALIDACION --> " + str(ds_validacion.shape[0]) + " x " + str(ds_validacion.shape[1]))

        # Con las siguientes 3 líneas dejo sólo 1 true y todo lo demás false en el dataframe de test. Asi en la matriz de confusion validamos si el sistema no añade falsos positivos
        # aux = ds_test[ds_test.TARGET == True]
        # ds_test = ds_test[ds_test.TARGET == False]
        # ds_test=ds_test.append(aux.iloc[0])

        print("Separamos FEATURES y TARGETS, de los 3 dataframes...")
        ds_train_f = ds_train.drop('TARGET', axis=1).to_numpy()
        ds_train_t = ds_train[['TARGET']].to_numpy().ravel()
        ds_test_f = ds_test.drop('TARGET', axis=1).to_numpy()
        ds_test_t = ds_test[['TARGET']].to_numpy().ravel()
        ds_validac_f = ds_validacion.drop('TARGET', axis=1).to_numpy()
        ds_validac_t = ds_validacion[['TARGET']].to_numpy().ravel()

        print("---------------- MODELOS con varias configuraciones (hiperparametros) --------")

        print("MODELOS: " + "https://scikit-learn.org/stable/supervised_learning.html")
        print("EVALUACION de los modelos con: " + "https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics")
        print("EVALUACION con curva precision-recall: " + "https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html")


        nombreModelo = "svc"
        pathModelo = dirModelos + str(id_subgrupo) + "_" + nombreModelo + ".modelo"
        modelo = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False,
                     tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                     decision_function_shape='ovr', break_ties=False, random_state=None)
        modelo_grid_mejores_parametros = ejecutarModeloyGuardarlo(nombreModelo, modelo, pathModelo, ds_train_f, ds_train_t, False, modoDebug)
        area_bajo_roc = cargarModeloyUsarlo(dirModelos, pathModelo, ds_test_f, ds_test_t, modoDebug)
        print(type(area_bajo_roc))
        if area_bajo_roc > ganador_area_bajo_roc:
           ganador_area_bajo_roc = area_bajo_roc
           ganador_nombreModelo = nombreModelo
           ganador_grid_mejores_parametros = modelo_grid_mejores_parametros


        nombreModelo = "logreg"
        pathModelo = dirModelos + str(id_subgrupo) + "_" + nombreModelo + ".modelo"
        modelo = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr')
        modelo_grid_mejores_parametros = ejecutarModeloyGuardarlo(nombreModelo, modelo, pathModelo, ds_train_f, ds_train_t, False, modoDebug)
        area_bajo_roc = cargarModeloyUsarlo(dirModelos, pathModelo, ds_test_f, ds_test_t, modoDebug)
        print(type(area_bajo_roc))
        if area_bajo_roc > ganador_area_bajo_roc:
            ganador_area_bajo_roc = area_bajo_roc
            ganador_nombreModelo = nombreModelo
            ganador_grid_mejores_parametros = modelo_grid_mejores_parametros


        nombreModelo = "rf"
        pathModelo = dirModelos + str(id_subgrupo) + "_" + nombreModelo + ".modelo"
        modelo = RandomForestClassifier(n_estimators=50, max_depth=2, random_state=0)
        modelo_grid_mejores_parametros = ejecutarModeloyGuardarlo(nombreModelo, modelo, pathModelo, ds_train_f, ds_train_t, False, modoDebug)
        area_bajo_roc = cargarModeloyUsarlo(dirModelos, pathModelo, ds_test_f, ds_test_t, modoDebug)
        print(type(area_bajo_roc))
        if area_bajo_roc > ganador_area_bajo_roc:
            ganador_area_bajo_roc = area_bajo_roc
            ganador_nombreModelo = nombreModelo
            ganador_grid_mejores_parametros = modelo_grid_mejores_parametros


        nombreModelo = "nn"
        pathModelo = dirModelos + str(id_subgrupo) + "_" + nombreModelo + ".modelo"
        modelo = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        modelo_grid_mejores_parametros = ejecutarModeloyGuardarlo(nombreModelo, modelo, pathModelo, ds_train_f, ds_train_t, False, modoDebug)
        area_bajo_roc = cargarModeloyUsarlo(dirModelos, pathModelo, ds_test_f, ds_test_t, modoDebug)
        print(type(area_bajo_roc))
        if area_bajo_roc > ganador_area_bajo_roc:
            ganador_area_bajo_roc = area_bajo_roc
            ganador_nombreModelo = nombreModelo
            ganador_grid_mejores_parametros = modelo_grid_mejores_parametros


        ####### HYPERPARAMETROS: GRID de parametros #######
        print("HYPERPARAMETROS - URL: https://scikit-learn.org/stable/modules/grid_search.html")

        nombreModelo = "svc_grid"
        pathModelo = dirModelos + str(id_subgrupo) + "_" + nombreModelo + ".modelo"
        modelo_base = svm.SVC()
        hiperparametros = [{'kernel': ['rbf'], 'C': [1, 10, 100]}, {'kernel': ['linear'], 'C': [1, 10, 100]}]
        modelos_grid = GridSearchCV(modelo_base, hiperparametros, n_jobs=-1, refit=True, cv=5, pre_dispatch='2*n_jobs', return_train_score=False)
        modelo_grid_mejores_parametros = ejecutarModeloyGuardarlo(nombreModelo, modelos_grid, pathModelo, ds_train_f, ds_train_t, True, modoDebug)
        area_bajo_roc = cargarModeloyUsarlo(dirModelos, pathModelo, ds_test_f, ds_test_t, modoDebug)
        print(type(area_bajo_roc))
        if area_bajo_roc > ganador_area_bajo_roc:
            ganador_area_bajo_roc = area_bajo_roc
            ganador_nombreModelo = nombreModelo
            ganador_grid_mejores_parametros = modelo_grid_mejores_parametros


        nombreModelo = "logreg_grid"
        pathModelo = dirModelos + str(id_subgrupo) + "_" + nombreModelo + ".modelo"
        modelo_base = LogisticRegression()
        hiperparametros = dict(C=np.logspace(0, 4, 10), penalty=['l2'])
        modelos_grid = GridSearchCV(modelo_base, hiperparametros, n_jobs=-1, refit=True, cv=5, pre_dispatch='2*n_jobs', return_train_score=False)
        modelo_grid_mejores_parametros = ejecutarModeloyGuardarlo(nombreModelo, modelos_grid, pathModelo, ds_train_f, ds_train_t, True, modoDebug)
        area_bajo_roc = cargarModeloyUsarlo(dirModelos, pathModelo, ds_test_f, ds_test_t, modoDebug)
        print(type(area_bajo_roc))
        if area_bajo_roc > ganador_area_bajo_roc:
            ganador_area_bajo_roc = area_bajo_roc
            ganador_nombreModelo = nombreModelo
            ganador_grid_mejores_parametros = modelo_grid_mejores_parametros


        nombreModelo = "rf_grid"
        pathModelo = dirModelos + str(id_subgrupo) + "_" + nombreModelo + ".modelo"
        modelo_base = CalibratedClassifierCV(base_estimator=RandomForestClassifier(n_estimators=50))
        hiperparametros = {'base_estimator__max_depth': [2, 5, 8, 13, 17, 25]}
        modelos_grid = GridSearchCV(modelo_base, hiperparametros, n_jobs=-1, refit=True, cv=5, pre_dispatch='2*n_jobs', return_train_score=False)
        modelo_grid_mejores_parametros = ejecutarModeloyGuardarlo(nombreModelo, modelos_grid, pathModelo, ds_train_f, ds_train_t, True, modoDebug)
        area_bajo_roc = cargarModeloyUsarlo(dirModelos, pathModelo, ds_test_f, ds_test_t, modoDebug)
        print(type(area_bajo_roc))
        if area_bajo_roc > ganador_area_bajo_roc:
            ganador_area_bajo_roc = area_bajo_roc
            ganador_nombreModelo = nombreModelo
            ganador_grid_mejores_parametros = modelo_grid_mejores_parametros


        print("********* GANADOR de subgrupo " + id_subgrupo + " *************")
        print("Modelo ganador es " + ganador_nombreModelo + " con un area_bajo_ROC de " + str(round(ganador_area_bajo_roc, 4)) +" y con estos hiperparametros: ")
        print(ganador_grid_mejores_parametros)
        pathModeloGanadorDeSubgrupo = dirModelos + str(id_subgrupo) + "_ganador" + ".modelo"


############################################################
print("------------ FIN ----------------")

