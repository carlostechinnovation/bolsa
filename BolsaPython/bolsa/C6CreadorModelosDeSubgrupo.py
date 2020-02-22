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

np.random.seed(12345)

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
granProbTargetUno = 60 # De todos los target=1, nos quedaremos con los granProbTargetUno (en tanto por cien) MAS probables. Un valor de 100 o mayor anula este parámetro

######### ID de subgrupo #######
partes = dir_subgrupo.split("/")
id_subgrupo = ""
for parte in partes:
    if (parte != ''):
        id_subgrupo = parte

########### Rutas #########
pathCsvCompleto = dir_subgrupo + "COMPLETO.csv"
pathCsvReducido = dir_subgrupo + "REDUCIDO.csv"
pathCsvPredichos = dir_subgrupo + "TARGETS_PREDICHOS.csv"
pathCsvPredichosIndices = dir_subgrupo + "TARGETS_PREDICHOS.csv_indices"
pathCsvFinalFuturo = dir_subgrupo + desplazamientoAntiguedad + "_" + id_subgrupo + "_COMPLETO_PREDICCION.csv"
dir_subgrupo_img = dir_subgrupo + "img/"
umbralProbTargetTrue = float("0.5")

print("dir_subgrupo: %s" % dir_subgrupo)
print("modoTiempo: %s" % modoTiempo)
print("desplazamientoAntiguedad: %s" % desplazamientoAntiguedad)
print("pathCsvReducido: %s" % pathCsvReducido)
print("dir_subgrupo_img = %s" % dir_subgrupo_img)
print("umbralProbTargetTrue = " + str(umbralProbTargetTrue))


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

    inputFeaturesyTarget = pd.read_csv(pathCsvReducido, index_col = 0, sep='|')  #La columna 0 contiene el indice
    print("inputFeaturesyTarget: " + str(inputFeaturesyTarget.shape[0]) + " x " + str(inputFeaturesyTarget.shape[1]))

    print("BALANCEAR los casos positivos y negativos, haciendo downsampling de la clase mayoritaria...")
    print("URL: https://elitedatascience.com/imbalanced-classes")
    ift_mayoritaria = inputFeaturesyTarget[inputFeaturesyTarget.TARGET == False]  # En este caso los mayoritarios son los False
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
        ift_juntas = pd.concat([ift_mayoritaria.reset_index(drop=True), ift_minoritaria.reset_index(drop=True)], axis=0)  # Row bind
        indices_juntos = ift_mayoritaria.index.append(ift_minoritaria.index)  # Row bind
        ift_juntas.set_index(indices_juntos, inplace=True)
        print("Las clases juntas son:")
        print("ift_juntas:" + str(ift_juntas.shape[0]) + " x " + str(ift_juntas.shape[1]))

        print("DIVIDIR EL DATASET DE ENTRADA EN 3 PARTES: TRAIN (50%), TEST (25%), VALIDACION (25%)...")
        ds_train, ds_test, ds_validacion = np.split(ift_juntas.sample(frac=1), [int(0.5 * len(ift_juntas)), int(0.75 * len(ift_juntas))])
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
        print("Resampling con SMOTE del vector de TRAINING (pero no a TEST ni a VALIDATION) según: " + "https://machinelearningmastery.com/combine-oversampling-and-undersampling-for-imbalanced-classification/")
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
        print("Accuracy: " + str(modelo_loaded.score(ds_test_f, ds_test_t)))
        print("Recall: " + str(recall_score(ds_test_t, modelo_loaded.predict(ds_test_f))))
        print("Validation Results")
        print("Accuracy: " + str(modelo_loaded.score(ds_validac_f, ds_validac_t)))
        print("Recall: " + str(recall_score(ds_validac_t, modelo_loaded.predict(ds_validac_f))))

elif (modoTiempo == "futuro" and pathCsvReducido.endswith('.csv') and os.path.isfile(pathCsvReducido) and os.stat(
        pathCsvReducido).st_size > 0):
    inputFeaturesyTarget = pd.read_csv(pathCsvReducido, index_col=0, sep='|')
    print("inputFeaturesyTarget: " + str(inputFeaturesyTarget.shape[0]) + " x " + str(inputFeaturesyTarget.shape[1]))
    print("La columna TARGET que haya en el CSV de entrada no la queremos (es un NULL o False, por defecto), porque la vamos a PREDECIR...")
    inputFeatures = inputFeaturesyTarget.drop('TARGET', axis=1)
    print(inputFeatures.head())
    print("inputFeatures: " + str(inputFeatures.shape[0]) + " x " + str(inputFeatures.shape[1]))

    print("MISSING VALUES (FILAS) - Borramos las FILAS que tengan 1 o mas valores NaN porque son huecos que no deberían estar...")
    inputFeatures_sinnulos = inputFeatures.dropna(axis=0, how='any')  # Borrar FILA si ALGUNO sus valores tienen NaN

    dir_modelo_predictor_ganador = dir_subgrupo.replace("futuro", "pasado")  #Siempre cojo el modelo entrenado en el pasado
    for file in os.listdir(dir_modelo_predictor_ganador):
        if file.endswith("ganador"):
            path_modelo_predictor_ganador = os.path.join(dir_modelo_predictor_ganador, file)

    print("Cargar modelo PREDICTOR ganador (de la carpeta del pasado, SI EXISTE): " + path_modelo_predictor_ganador)
    if os.path.isfile(path_modelo_predictor_ganador):

        modelo_predictor_ganador = pickle.load(open(path_modelo_predictor_ganador, 'rb'))

        print("Predecir:")
        targets_predichos = modelo_predictor_ganador.predict(inputFeatures_sinnulos)
        print("Numero de targets_predichos: " + str(len(targets_predichos)) + " con numero de TRUEs = " + str(np.sum(targets_predichos, where=["True"])))

        # probabilities
        probs = pd.DataFrame(data=modelo_predictor_ganador.predict_proba(inputFeatures_sinnulos), index=inputFeatures_sinnulos.index)

        # UMBRAL MENOS PROBABLES CON TARGET=1. Cuando el target es 1, se guarda su probabilidad
        print(probs.columns)
        probabilidadesEnTargetUnoPeq = probs.iloc[:,1]  # Cogemos solo la segunda columna: prob de que sea target=1
        probabilidadesEnTargetUnoPeq2 = probabilidadesEnTargetUnoPeq.apply(lambda x: x if(x >= umbralProbTargetTrue) else np.nan)
        probabilidadesEnTargetUnoPeq3 = probabilidadesEnTargetUnoPeq2[np.isnan(probabilidadesEnTargetUnoPeq2[:]) == False] # Cogemos todos los no nulos
        probabilidadesEnTargetUnoPeq4 = probabilidadesEnTargetUnoPeq3.sort_values(ascending=False)
        numfilasSeleccionadas = int(granProbTargetUno * probabilidadesEnTargetUnoPeq4.shape[0] / 100)  #Como están ordenadas en descendente, cojo estas NUM primeras filas
        targets_predichosCorregidos_probs = probabilidadesEnTargetUnoPeq4[0:(numfilasSeleccionadas-1)]
        targets_predichosCorregidos = targets_predichosCorregidos_probs.apply(lambda x: True)

        print("Guardando targets PREDICHOS en: " + pathCsvPredichos)
        df_predichos = targets_predichosCorregidos.to_frame()
        df_predichos.columns = ['TARGET_PREDICHO']
        df_predichos.to_csv(pathCsvPredichos, index=False, sep='|')  # Capa 6 - Salida (para el validador, sin indice)

        df_predichos_probs = targets_predichosCorregidos_probs.to_frame()
        df_predichos_probs.columns = ['TARGET_PREDICHO_PROB']
        df_predichos_probs.to_csv(pathCsvPredichos + "_humano", index=True, sep='|')  # Capa 6 - Salida (para el humano)

        ############### RECONSTRUCCION DEL CSV FINAL IMPORTANTE, viendo los ficheros de indices #################
        print("Partiendo de COMPLETO.csv llevamos la cuenta de los indices pasando por REDUCIDO.csv y por TARGETS_PREDICHOS.csv para generar el CSV final...")
        df_completo = pd.read_csv(pathCsvCompleto, sep='|')  #Capa 5 - Entrada

        print("df_completo: " + str(df_completo.shape[0]) + " x " + str(df_completo.shape[1]))
        print("df_predichos: " + str(df_predichos.shape[0]) + " x " + str(df_predichos.shape[1]))
        print("df_predichos_probs: " + str(df_predichos_probs.shape[0]) + " x " + str(df_predichos_probs.shape[1]))

        print("Juntar COMPLETO con TARGETS PREDICHOS... ")
        df_juntos_1 = pd.concat([df_completo, df_predichos], axis=1)
        df_juntos_2 = pd.concat([df_juntos_1, df_predichos_probs], axis=1)

        df_juntos_2['TARGET_PREDICHO'] = (df_juntos_2['TARGET_PREDICHO'] * 1).astype('Int64')  # Convertir de boolean a int64, manteniendo los nulos

        print("Guardando: " + pathCsvFinalFuturo)
        df_juntos_2.to_csv(pathCsvFinalFuturo, index=False, sep='|')


    else:
        print("No existe el modelo predictor del pasado que necesitamos (" + path_modelo_predictor_ganador + "). Por tanto, no predecimos.")


else:
    print("Los parametros de entrada son incorrectos o el CSV no existe o esta vacio!!")

############################################################
print("------------ FIN de capa 6----------------")