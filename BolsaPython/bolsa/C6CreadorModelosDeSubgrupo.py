import sys
import pandas as pd
import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from numpy import mean
from scipy.stats import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.fixes import loguniform

np.random.seed(12345)
from pathlib import Path
from sklearn import svm
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, recall_score, make_scorer
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from shutil import copyfile
import os.path
from imblearn.over_sampling import SMOTE
from sklearn.metrics import make_scorer, accuracy_score

print("-------------------------------------------------------------------------------------------------")
print("---- CAPA 6 - Crear almacenar y evaluar varios modelos (para cada subgrupo) -------")
print("Tipo de problema: CLASIFICACION BINOMIAL (target es boolean)")

print("PARAMETROS: ")
dir_subgrupo = sys.argv[1]
modoTiempo = sys.argv[2]
desplazamientoAntiguedad = sys.argv[3]
modoDebug = False  # En modo debug se pintan los dibujos. En otro caso, se evita calculo innecesario
umbralCasosSuficientesClasePositiva = 100
RecallOAreaROC = False  # True si se toma recall para seleccionar el modelo. False si se toma Area Bajo ROC
pequenaProbTargetUno = 100  # De todos los target=1, nos quedaremos con los pequenaProbTargetUno (en tanto por cien) MENOS probables. Un valor de 100 o mayor anula este parámetro
granProbTargetUno = 100 # De todos los target=1, nos quedaremos con los granProbTargetUno (en tanto por cien) MAS probables. Un valor de 100 o mayor anula este parámetro

######### ID de subgrupo #######
partes = dir_subgrupo.split("/")
id_subgrupo = ""
for parte in partes:
    if (parte != ''):
        id_subgrupo = parte

########### Rutas #########
pathCsvCompleto = dir_subgrupo + "COMPLETO.csv"
pathCsvReducido = dir_subgrupo + "REDUCIDO.csv"
pathCsvReducidoIndices = dir_subgrupo + "REDUCIDO.csv_indices"
pathCsvPredichos = dir_subgrupo + "TARGETS_PREDICHOS.csv"
pathCsvPredichosIndices = dir_subgrupo + "TARGETS_PREDICHOS.csv_indices"
pathCsvFinalFuturo = dir_subgrupo + desplazamientoAntiguedad + "_" + id_subgrupo + "_COMPLETO_PREDICCION.csv"
dir_subgrupo_img = dir_subgrupo + "img/"

print("dir_subgrupo: %s" % dir_subgrupo)
print("modoTiempo: %s" % modoTiempo)
print("desplazamientoAntiguedad: %s" % desplazamientoAntiguedad)
print("pathCsvReducido: %s" % pathCsvReducido)
print("dir_subgrupo_img = %s" % dir_subgrupo_img)


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


def cargarModeloyUsarlo(dir_subgrupo_img, pathModelo, ds_test_f, ds_test_t, modoDebug):
    modelo_loaded = pickle.load(open(pathModelo, 'rb'))
    ds_test_t_pred = modelo_loaded.predict(
        ds_test_f)  # PREDICCION de los targets de TEST (los compararemos con los que tenemos)
    area_bajo_roc = roc_auc_score(ds_test_t, ds_test_t_pred)

    print(nombreModelo + ".roc_auc_score = " + str(round(area_bajo_roc, 4)))
    average_precision = average_precision_score(ds_test_t, ds_test_t_pred)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    recall = recall_score(ds_test_t, ds_test_t_pred, average='binary', pos_label=1)
    print('Average recall score: {0:0.2f}'.format(recall))

    if modoDebug:
        print("Curva ROC...")
        # EVALUACION DE MODELOS - Curva ROC: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
        fpr_modelo, tpr_modelo, _ = roc_curve(ds_test_t, ds_test_t_pred)
        path_dibujo = dir_subgrupo_img + nombreModelo + "_roc.png"
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
        path_dibujo = dir_subgrupo_img + nombreModelo + "_matriz_conf.png"
        disp = plot_confusion_matrix(modelo_loaded, ds_test_f, ds_test_t, cmap=plt.cm.Blues, normalize=None)
        disp.ax_.set_title(nombreModelo)
        print(disp.confusion_matrix)
        plt.savefig(path_dibujo, bbox_inches='tight')
        # Limpiando dibujo:
        plt.clf()
        plt.cla()
        plt.close()

    if RecallOAreaROC:
        return recall
    else:
        return area_bajo_roc


################# MAIN ########################################
# GANADOR DEL SUBGRUPO (acumuladores)
ganador_nombreModelo = "NINGUNO"
ganador_recall_o_ROC = 0
ganador_grid_mejores_parametros = []

if (modoTiempo == "pasado" and pathCsvReducido.endswith('.csv') and os.path.isfile(pathCsvReducido) and os.stat(
        pathCsvReducido).st_size > 0):

    inputFeaturesyTarget = pd.read_csv(pathCsvReducido, sep='|')
    print("inputFeaturesyTarget: " + str(inputFeaturesyTarget.shape[0]) + " x " + str(inputFeaturesyTarget.shape[1]))

    print("BALANCEAR los casos positivos y negativos, haciendo downsampling de la clase mayoritaria...")
    print("URL: https://elitedatascience.com/imbalanced-classes")
    ift_mayoritaria = inputFeaturesyTarget[
        inputFeaturesyTarget.TARGET == False]  # En este caso los mayoritarios son los False
    ift_minoritaria = inputFeaturesyTarget[inputFeaturesyTarget.TARGET == True]
    print("ift_mayoritaria:" + str(ift_mayoritaria.shape[0]) + " x " + str(ift_mayoritaria.shape[1]))
    print("ift_minoritaria:" + str(ift_minoritaria.shape[0]) + " x " + str(ift_minoritaria.shape[1]))

    print("Tasa de desbalanceo entre clases = " + str(ift_mayoritaria.shape[0]) + "/" + str(
        ift_minoritaria.shape[0]) + " = " + str(ift_mayoritaria.shape[0] / ift_minoritaria.shape[0]))
    num_muestras_minoria = ift_minoritaria.shape[0]

    casosInsuficientes = (num_muestras_minoria < umbralCasosSuficientesClasePositiva)
    if (casosInsuficientes):
        print("Numero de casos en clase minoritaria es INSUFICIENTE. Así que abandonamos este dataset y seguimos")

    else:

        print("num_muestras_minoria: " + str(num_muestras_minoria))

        # CORRECCIÓN LUIS: no debo hacer resample aquí, sino sólo en los datos de training
        # ift_mayoritaria_downsampled = resample(ift_mayoritaria, replace=False, n_samples=num_muestras_minoria,
        #                                        random_state=123)
        ift_mayoritaria_downsampled = ift_mayoritaria

        ift_balanceadas = pd.concat(
            [ift_mayoritaria_downsampled, ift_minoritaria])  # Juntar ambas clases ya BALANCEADAS

        # Desordeno la matriz, para que no se entrene/test/valide por fechas, o por patrones dependientes del tiempo
        ift_balanceadas = ift_balanceadas.sample(frac=1).reset_index(drop=True)

        print("Las clases ya están balanceadas:")
        print("ift_mayoritaria_downsampled:" + str(ift_mayoritaria_downsampled.shape[0]) + " x " + str(ift_mayoritaria_downsampled.shape[1]))
        print("ift_minoritaria:" + str(ift_minoritaria.shape[0]) + " x " + str(ift_minoritaria.shape[1]))
        print("ift_balanceadas:" + str(ift_balanceadas.shape[0]) + " x " + str(ift_balanceadas.shape[1]))

        print("DIVIDIR EL DATASET DE ENTRADA EN 3 PARTES: TRAIN (50%), TEST (25%), VALIDACION (25%)...")
        ds_train, ds_test, ds_validacion = np.split(ift_balanceadas.sample(frac=1),
                                                    [int(.5 * len(ift_balanceadas)), int(.75 * len(ift_balanceadas))])
        print("TRAIN --> " + str(ds_train.shape[0]) + " x " + str(ds_train.shape[1]))
        print("TEST --> " + str(ds_test.shape[0]) + " x " + str(ds_test.shape[1]))
        print("VALIDACION --> " + str(ds_validacion.shape[0]) + " x " + str(ds_validacion.shape[1]))

        # Con las siguientes 3 líneas dejo sólo 1 true y lo demás false en el dataframe de test. Asi en la matriz de confusion validamos si el sistema no añade falsos positivos
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

        # SMOTE (Over and undersampling on the train data):
        print(
            "Resampling con SMOTE del vector de training según: " + "https://machinelearningmastery.com/combine-oversampling-and-undersampling-for-imbalanced-classification/")
        print("---------------- RESAMPLING con SMOTE --------")
        resample = SMOTEENN()
        print("SMOTE antes: %d" % ds_train_f.shape[0])
        ds_train_f, ds_train_t = resample.fit_sample(ds_train_f, ds_train_t)
        print("SMOTE después: %d" % ds_train_f.shape[0])

        print("---------------- MODELOS con varias configuraciones (hiperparametros) --------")

        print("MODELOS: " + "https://scikit-learn.org/stable/supervised_learning.html")
        print(
            "EVALUACION de los modelos con: " + "https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics")
        print(
            "EVALUACION con curva precision-recall: " + "https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html")

        # nombreModelo = "svc"
        # pathModelo = dir_subgrupo + nombreModelo + ".modelo"
        # modelo = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True,
        #              tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
        #              decision_function_shape='ovr', break_ties=False, random_state=None)
        # modelo_grid_mejores_parametros = ejecutarModeloyGuardarlo(nombreModelo, modelo, pathModelo, ds_train_f, ds_train_t, False, modoDebug)
        # recall_o_ROC = cargarModeloyUsarlo(dir_subgrupo_img, pathModelo, ds_test_f, ds_test_t, modoDebug)
        # print(type(recall_o_ROC))
        # if recall_o_ROC > ganador_recall_o_ROC:
        #    ganador_recall_o_ROC = recall_o_ROC
        #    ganador_nombreModelo = nombreModelo
        #    ganador_grid_mejores_parametros = modelo_grid_mejores_parametros
        #
        #
        # nombreModelo = "logreg"
        # pathModelo = dir_subgrupo + nombreModelo + ".modelo"
        # modelo = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr')
        # modelo_grid_mejores_parametros = ejecutarModeloyGuardarlo(nombreModelo, modelo, pathModelo, ds_train_f, ds_train_t, False, modoDebug)
        # recall_o_ROC = cargarModeloyUsarlo(dir_subgrupo_img, pathModelo, ds_test_f, ds_test_t, modoDebug)
        # print(type(recall_o_ROC))
        # if recall_o_ROC > ganador_recall_o_ROC:
        #     ganador_recall_o_ROC = recall_o_ROC
        #     ganador_nombreModelo = nombreModelo
        #     ganador_grid_mejores_parametros = modelo_grid_mejores_parametros

        # nombreModelo = "rf"
        # pathModelo = dir_subgrupo + nombreModelo + ".modelo"
        # modelo = RandomForestClassifier(n_estimators=50, max_depth=2, random_state=0)
        # modelo_grid_mejores_parametros = ejecutarModeloyGuardarlo(nombreModelo, modelo, pathModelo, ds_train_f, ds_train_t, False, modoDebug)
        # recall_o_ROC = cargarModeloyUsarlo(dir_subgrupo_img, pathModelo, ds_test_f, ds_test_t, modoDebug)
        # print(type(recall_o_ROC))
        # if recall_o_ROC > ganador_recall_o_ROC:
        #     ganador_recall_o_ROC = recall_o_ROC
        #     ganador_nombreModelo = nombreModelo
        #     ganador_grid_mejores_parametros = modelo_grid_mejores_parametros

        # nombreModelo = "nn"
        # pathModelo = dir_subgrupo + nombreModelo + ".modelo"
        # modelo = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        # modelo_grid_mejores_parametros = ejecutarModeloyGuardarlo(nombreModelo, modelo, pathModelo, ds_train_f, ds_train_t, False, modoDebug)
        # recall_o_ROC = cargarModeloyUsarlo(dir_subgrupo_img, pathModelo, ds_test_f, ds_test_t, modoDebug)
        # print(type(recall_o_ROC))
        # if recall_o_ROC > ganador_recall_o_ROC:
        #     ganador_recall_o_ROC = recall_o_ROC
        #     ganador_nombreModelo = nombreModelo
        #     ganador_grid_mejores_parametros = modelo_grid_mejores_parametros

        ####### HYPERPARAMETROS: GRID de parametros #######
        print("HYPERPARAMETROS - URL: https://scikit-learn.org/stable/modules/grid_search.html")

        # nombreModelo = "svc_grid"
        # pathModelo = dir_subgrupo + nombreModelo + ".modelo"
        # modelo_base = svm.SVC()
        # hiperparametros = [{'C':[1,3,5,10],'gamma':[1], 'kernel':['rbf']}]
        # modelos_grid = GridSearchCV(modelo_base, hiperparametros, n_jobs=-1, refit=True, cv=5, pre_dispatch='2*n_jobs', return_train_score=False)
        # modelo_grid_mejores_parametros = ejecutarModeloyGuardarlo(nombreModelo, modelos_grid, pathModelo, ds_train_f, ds_train_t, True, modoDebug)
        # recall_o_ROC = cargarModeloyUsarlo(dir_subgrupo_img, pathModelo, ds_test_f, ds_test_t, modoDebug)
        # print(type(recall_o_ROC))
        # if recall_o_ROC > ganador_recall_o_ROC:
        #     ganador_recall_o_ROC = recall_o_ROC
        #     ganador_nombreModelo = nombreModelo
        #     ganador_grid_mejores_parametros = modelo_grid_mejores_parametros
        #
        #
        # nombreModelo = "logreg_grid"
        # pathModelo = dir_subgrupo + nombreModelo + ".modelo"
        # modelo_base = LogisticRegression()
        # hiperparametros = dict(C=np.logspace(0, 4, 10), penalty=['l2'])
        # modelos_grid = GridSearchCV(modelo_base, hiperparametros, n_jobs=-1, refit=True, cv=5, pre_dispatch='2*n_jobs', return_train_score=False)
        # modelo_grid_mejores_parametros = ejecutarModeloyGuardarlo(nombreModelo, modelos_grid, pathModelo, ds_train_f, ds_train_t, True, modoDebug)
        # recall_o_ROC = cargarModeloyUsarlo(dir_subgrupo_img, pathModelo, ds_test_f, ds_test_t, modoDebug)
        # print(type(recall_o_ROC))
        # if recall_o_ROC > ganador_recall_o_ROC:
        #     ganador_recall_o_ROC = recall_o_ROC
        #     ganador_nombreModelo = nombreModelo
        #     ganador_grid_mejores_parametros = modelo_grid_mejores_parametros

        nombreModelo = "rf_grid"
        pathModelo = dir_subgrupo + nombreModelo + ".modelo"
        modelo_base = CalibratedClassifierCV(base_estimator=RandomForestClassifier())
        hiperparametros = {'base_estimator__n_estimators': [14, 20, 40, 70],
                           'base_estimator__max_features': ['log2', 'sqrt', 'auto'],
                           'base_estimator__criterion': ['entropy', 'gini'],
                           'base_estimator__max_depth': [15, 25, 85, 300, 1000],
                           'base_estimator__min_samples_split': [3],
                           'base_estimator__min_samples_leaf': [1]}
        modelos_grid = GridSearchCV(modelo_base, hiperparametros, scoring='recall', n_jobs=-1, refit=True,
                                    return_train_score=False)
        modelo_grid_mejores_parametros = ejecutarModeloyGuardarlo(nombreModelo, modelos_grid, pathModelo, ds_train_f,
                                                                  ds_train_t, True, modoDebug)
        recall_o_ROC = cargarModeloyUsarlo(dir_subgrupo_img, pathModelo, ds_test_f, ds_test_t, modoDebug)
        print(type(recall_o_ROC))
        if recall_o_ROC > ganador_recall_o_ROC:
            ganador_recall_o_ROC = recall_o_ROC
            ganador_nombreModelo = nombreModelo
            ganador_grid_mejores_parametros = modelo_grid_mejores_parametros

        print("********* GANADOR de subgrupo *************")
        print("Modelo ganador es " + ganador_nombreModelo + " con un recall (o area bajo ROC) de " + str(
            round(ganador_recall_o_ROC, 4)) + " y con estos hiperparametros: ")
        print(ganador_grid_mejores_parametros)
        pathModeloGanadorDeSubgrupoOrigen = dir_subgrupo + ganador_nombreModelo + ".modelo"
        pathModeloGanadorDeSubgrupoDestino = pathModeloGanadorDeSubgrupoOrigen + "_ganador"
        copyfile(pathModeloGanadorDeSubgrupoOrigen, pathModeloGanadorDeSubgrupoDestino)
        print("Modelo ganador guardado en: " + pathModeloGanadorDeSubgrupoDestino)

        modelo_loaded = pickle.load(open(pathModelo, 'rb'))
        ds_test_t_pred = modelo_loaded.predict(ds_test_f)

        print("LOS RESULTADOS DE VALIDACION Y TEST DEBERÍAN SER SIMILARES. SI NO, ESTARIÁMOS COMETIENDO ERRORES...")
        print("\nTest Results")
        print("Accuracy: "+str(modelo_loaded.score(ds_test_f, ds_test_t)))
        print("Recall: "+str(recall_score(ds_test_t, modelo_loaded.predict(ds_test_f))))
        print("Validation Results")
        print("Accuracy: "+str(modelo_loaded.score(ds_validac_f, ds_validac_t)))
        print("Recall: "+str(recall_score(ds_validac_t, modelo_loaded.predict(ds_validac_f))))

elif (modoTiempo == "futuro" and pathCsvReducido.endswith('.csv') and os.path.isfile(pathCsvReducido) and os.stat(
        pathCsvReducido).st_size > 0):
    inputFeaturesyTarget = pd.read_csv(pathCsvReducido, sep='|')
    print("inputFeaturesyTarget: " + str(inputFeaturesyTarget.shape[0]) + " x " + str(inputFeaturesyTarget.shape[1]))
    print(
        "La columna TARGET que haya en el CSV de entrada no la queremos (es un NULL o False, por defecto), porque la vamos a PREDECIR...")
    inputFeatures = inputFeaturesyTarget.drop('TARGET', axis=1)
    print(inputFeatures.head())
    print("inputFeatures: " + str(inputFeatures.shape[0]) + " x " + str(inputFeatures.shape[1]))

    print("Capturamos el indice inicial de filas de entrada:")
    indiceFilasFuturasTransformadas = inputFeaturesyTarget.index.values

    print(
        "MISSING VALUES (FILAS) - Borramos las FILAS que tengan 1 o mas valores NaN porque son huecos que no deberían estar...")
    inputFeatures_sinnulos = inputFeatures.dropna(axis=0, how='any')  # Borrar FILA si ALGUNO sus valores tienen NaN
    indiceFilasFuturasTransformadas2 = inputFeatures_sinnulos.index.values
    inputFeatures_sinnulos = inputFeatures_sinnulos.to_numpy()
    print("inputFeatures_sinnulos (filas algun nulo borradas):" + str(inputFeatures_sinnulos.shape[0]) + " x " + str(
        inputFeatures_sinnulos.shape[1]))

    dir_modelo_predictor_ganador = dir_subgrupo.replace("futuro", "pasado")
    for file in os.listdir(dir_modelo_predictor_ganador):
        if file.endswith("ganador"):
            path_modelo_predictor_ganador = os.path.join(dir_modelo_predictor_ganador, file)

    print("Cargar modelo PREDICTOR ganador (de la carpeta del pasado, SI EXISTE): " + path_modelo_predictor_ganador)
    if os.path.isfile(path_modelo_predictor_ganador):

        modelo_predictor_ganador = pickle.load(open(path_modelo_predictor_ganador, 'rb'))

        print("Predecir:")
        targets_predichos = modelo_predictor_ganador.predict(inputFeatures_sinnulos)

        # probabilities
        probs = modelo_predictor_ganador.predict_proba(inputFeatures_sinnulos)

        # UMBRAL MENOS PROBABLES CON TARGET=1. Cuando el target es 1, se guarda su probabilidad
        probabilidadesEnTargetUnoPeq = []
        for row in probs:
            if row[0] < row[1]:
                # La probabilidad de un 1 es mayor que la de un 0
                probabilidadesEnTargetUnoPeq.append(row[1])

        # De todos los target=1, nos quedaremos con los pequenaProbTargetUno % MENOS probables
        probabilidadesEnTargetUnoPeq.sort(reverse=False)
        numeroElementosPeq = len(probabilidadesEnTargetUnoPeq)
        indiceUltimoElementoSeleccionadoPeq = int(pequenaProbTargetUno * numeroElementosPeq / 100)
        probabsSeleccionadasPeq = probabilidadesEnTargetUnoPeq[0:indiceUltimoElementoSeleccionadoPeq]
        probabUmbralPeq = probabsSeleccionadasPeq[-1]

        # UMBRAL MAS PROBABLES CON TARGET=1. Cuando el target es 1, se guarda su probabilidad
        probabilidadesEnTargetUnoGran = []
        for row in probs:
            if row[0] < row[1]:
                # La probabilidad de un 1 es mayor que la de un 0
                probabilidadesEnTargetUnoGran.append(row[1])

        # De todos los target=1, nos quedaremos con los pequenaProbTargetUno % MAS probables
        probabilidadesEnTargetUnoGran.sort(reverse=True)
        numeroElementosGran = len(probabilidadesEnTargetUnoGran)
        indiceUltimoElementoSeleccionadoGran = int(pequenaProbTargetUno * numeroElementosGran / 100)
        probabsSeleccionadasGran = probabilidadesEnTargetUnoGran[0:indiceUltimoElementoSeleccionadoGran]
        probabUmbralGran = probabsSeleccionadasGran[-1]

        targets_predichosCorregidos = []
        probsSeleccionadasEnUno = []
        print("Umbral de probabilidad pequeña para target = 1: " + str(probabUmbralPeq))
        print("Umbral de probabilidad grande para target = 1: " + str(probabUmbralGran))
        for i in range(probs.shape[0]):
            prob = probs[i][1]
            # True si es poco probable cuando target=1
            if probs[i][0] < probs[i][1] and probabUmbralPeq >= prob >= probabUmbralGran:
                targets_predichosCorregidos.append("True")
                probsSeleccionadasEnUno.append(prob)
            else:
                targets_predichosCorregidos.append("False")

        print("Numero de targets_predichos: " + str(len(targets_predichos)) + " con numero de TRUEs = " + str(
            np.sum(targets_predichos, where=["True"])))
        print("Numero de targets_predichos corregidos: " + str(
            len(targets_predichosCorregidos)) + " con numero de TRUEs = " + str(len(probsSeleccionadasEnUno)))
        print("Probabilidades de los target=1 seleccionados: ")
        print(probsSeleccionadasEnUno)
        print("Guardando targets PREDICHOS en: " + pathCsvPredichos)
        pathCsvPredichos_df = pd.DataFrame(data=targets_predichosCorregidos, columns=['TARGET'])
        pathCsvPredichos_df.to_csv(pathCsvPredichos, index=False, sep='|')

        print("Guardando indices de filas de salida respecto de la entrada...")
        np.savetxt(pathCsvPredichos + "_indices", indiceFilasFuturasTransformadas2, delimiter="|", header="indice",
                   fmt='%f')

        ############### RECONSTRUCCION DEL CSV FINAL IMPORTANTE, viendo los ficheros de indices #################
        print(
            "Partiendo de COMPLETO.csv llevamos la cuenta de los indices pasando por REDUCIDO.csv y por TARGETS_PREDICHOS.csv para generar el CSV final...")
        df_completo = pd.read_csv(pathCsvCompleto, sep='|')
        df_reducido = pd.read_csv(pathCsvReducido, sep='|')
        df_reducido_indices = pd.read_csv(pathCsvReducidoIndices, sep='|').to_numpy()
        df_predichos = pd.read_csv(pathCsvPredichos, sep='|')
        df_predichos_indices = pd.read_csv(pathCsvPredichosIndices, sep='|').to_numpy()

        print("df_completo: " + str(df_completo.shape[0]) + " x " + str(df_completo.shape[1]))
        print("df_reducido: " + str(df_reducido.shape[0]) + " x " + str(df_reducido.shape[1]))
        print("df_reducido_indices: " + str(df_reducido_indices.shape[0]) + " x " + str(df_reducido_indices.shape[1]))
        print("df_predichos: " + str(df_predichos.shape[0]) + " x " + str(df_predichos.shape[1]))
        print(
            "df_predichos_indices: " + str(df_predichos_indices.shape[0]) + " x " + str(df_predichos_indices.shape[1]))

        print("COMPLETO_1 --> Del COMPLETO, cogemos solo las filas que se han usado para calcular el REDUCIDO...")
        df_completo_1 = df_completo.loc[df_reducido_indices[:, 0].tolist()].reset_index(drop=False, inplace=False).drop(
            'index', axis=1).drop('TARGET', axis=1)
        print("df_completo_1: " + str(df_completo_1.shape[0]) + " x " + str(df_completo_1.shape[1]))

        print("COMPLETO_2 --> Del COMPLETO_1, cogemos solo las filas que se han usado para calcular el PREDICHOS...")
        df_completo_2 = df_completo_1.loc[df_predichos_indices[:, 0].tolist()]
        print("df_completo_2: " + str(df_completo_2.shape[0]) + " x " + str(df_completo_2.shape[1]))

        print("Juntar COMPLETO con TARGETS PREDICHOS... ")
        df_juntos_1 = pd.concat([df_completo_2.reset_index(drop=True), df_predichos], axis=1)
        lado_derecho = df_juntos_1[['empresa', 'mercado', 'antiguedad', 'TARGET']]
        df_juntos_2 = pd.merge(df_completo, lado_derecho, how='left', on=['empresa', 'mercado', 'antiguedad'])
        df_juntos_2.rename(columns={'TARGET_x': 'TARGET_REAL', 'TARGET_y': 'TARGET_PREDICHO'}, inplace=True)

        df_juntos_2['TARGET_PREDICHO'] = (df_juntos_2['TARGET_PREDICHO'] * 1).astype(
            'Int64')  # Convertir de boolean a int64, manteniendo los nulos

        print("Guardando: " + pathCsvFinalFuturo)
        df_juntos_2.to_csv(pathCsvFinalFuturo, index=False, sep='|')


    else:
        print(
            "No existe el modelo predictor del pasado que necesitamos (" + path_modelo_predictor_ganador + "). Por tanto, no predecimos.")


else:
    print("Los parametros de entrada son incorrectos o el CSV no existe o esta vacio!!")

############################################################
print("------------ FIN de capa 6----------------")