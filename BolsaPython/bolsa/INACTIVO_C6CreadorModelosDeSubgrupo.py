import sys
import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np
from imblearn.combine import SMOTEENN
from numpy import mean
from sklearn.feature_selection import SelectKBest
from pathlib import Path
from sklearn.linear_model import LogisticRegression, SGDClassifier
import pickle
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV, cross_val_score
from shutil import copyfile
import os.path
from sklearn.tree import export_graphviz
from xgboost import XGBClassifier
from matplotlib import pyplot
import datetime
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import time
import matplotlib.patches as mpatches

np.random.seed(12345)

print("\n" + (datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " **** CAPA 6 - Crear almacenar y evaluar varios modelos (para cada subgrupo) ****")
print("Tipo de problema: CLASIFICACION DICOTOMICA (target es boolean)")

print("PARAMETROS: ")
dir_subgrupo = sys.argv[1]
modoTiempo = sys.argv[2]
desplazamientoAntiguedad = sys.argv[3]
pathFeaturesSeleccionadas = dir_subgrupo + "FEATURES_SELECCIONADAS.csv"
modoDebug = False  # En modo debug se pintan los dibujos. En otro caso, se evita calculo innecesario
umbralCasosSuficientesClasePositiva = 50
granProbTargetUno = 50  # De todos los target=1, nos quedaremos con los granProbTargetUno (en tanto por cien) MAS probables. Un valor de 100 o mayor anula este parámetro
balancearConSmoteSoloTrain = True
umbralFeaturesCorrelacionadas = 0.96  # Umbral aplicado para descartar features cuya correlacion sea mayor que él
umbralNecesarioCompensarDesbalanceo = 1  # Umbral de desbalanceo clase positiva/negativa. Si se supera, es necesario hacer oversampling de minoritaria (SMOTE) o undersampling de mayoritaria (borrar filas)
cv_todos = 10  # CROSS_VALIDATION: número de iteraciones. Sirve para evitar el overfitting
fraccion_train = 0.50  # Fracción de datos usada para entrenar
fraccion_test = 0.25  # Fracción de datos usada para testear (no es validación)
fraccion_valid = 1.00 - (fraccion_train + fraccion_test)

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
umbralProbTargetTrue = float("0.50")

print("dir_subgrupo: %s" % dir_subgrupo)
print("modoTiempo: %s" % modoTiempo)
print("desplazamientoAntiguedad: %s" % desplazamientoAntiguedad)
print("pathCsvReducido: %s" % pathCsvReducido)
print("dir_subgrupo_img = %s" % dir_subgrupo_img)
print("umbralProbTargetTrue = " + str(umbralProbTargetTrue))
print("balancearConSmoteSoloTrain = " + str(balancearConSmoteSoloTrain))
print("umbralFeaturesCorrelacionadas = " + str(umbralFeaturesCorrelacionadas))


################# FUNCIONES ########################################
def corr_drop(corr_m, factor=.9):
    """
    Drop correlated features maintaining the most relevant.

    Parameters
    ----------
    corr_m : pandas.DataFrame
        Correlation matrix
    factor : float
        Min correlation level

    Returns
    ----------
    pandas.DataFrame
        Correlation matrix only with most relevant features
    """
    global cm
    cm = corr_m
    # Get correlation score, as high as this score, more chances to be dropped.
    cum_corr = cm.applymap(abs).sum()

    def remove_corr():
        global cm
        for col in cm.columns:
            for ind in cm.index:
                if (ind in cm.columns) and (col in cm.index):
                    # Compare if are high correlated.
                    if (cm.loc[ind, col] > factor) and (ind != col):
                        cum = cum_corr[[ind, col]].sort_values(ascending=False)
                        cm.drop(cum.index[0], axis=0, inplace=True)
                        cm.drop(cum.index[0], axis=1, inplace=True)
                        # Do recursion until the last high correlated.
                        remove_corr()
        return cm

    return remove_corr()


def ejecutarModeloyGuardarlo(nombreModelo, modelo, pathModelo, ds_train_f, ds_train_t, ds_test_f, ds_test_t, feature_names, modeloEsGrid,
                             modoDebug, dir_subgrupo_img):
    print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " Ejecutando " + nombreModelo + " ...")
    out_grid_best_params = []

    param_parada_iteraciones = 10  # early_stopping_rounds: es el numero de iteraciones en las que ya no mejora el error diferencial train-test, evitando iterar tanto en XGBoost y reducir el overfitting
    eval_set = [(ds_train_f, ds_train_t), (ds_test_f, ds_test_t)]

    #-------- PINTAR EL ERROR DE OVERFITTING ---------------------------
    #-------------------URL: https://machinelearningmastery.com/avoid-overfitting-by-early-stopping-with-xgboost-in-python/
    #--- URL: https://xgboost.readthedocs.io/en/latest/parameter.html

    METODO_EVALUACION="map"  # map: Mean average Precision. aucpr: Area under the PR curve (peores resultados en precisión)
    # Explicacion: https://xgboost.readthedocs.io/en/latest/parameter.html

    # Con PARAMETROS PARA VER EL OVERFITTING
    modelo = modelo.fit(ds_train_f, ds_train_t, eval_metric=[METODO_EVALUACION], early_stopping_rounds=param_parada_iteraciones, eval_set=eval_set, verbose=False)  # ENTRENAMIENTO (TRAIN)

    # --------------- Pintar dibujo---------------------------------------------------------------
    y_pred = modelo.predict(ds_test_f)
    y_pred = y_pred.astype(float)
    predictions = [round(value) for value in y_pred]
    precision_para_medir_overfitting = precision_score(ds_test_t, predictions)
    print("Accuracy (PRECISION) para medir el overfitting: %.2f%%" % (precision_para_medir_overfitting * 100.0))
    results = modelo.evals_result()

    epochs = len(results['validation_0'][METODO_EVALUACION])
    x_axis = range(0, epochs)
    fig, ax = pyplot.subplots()
    ax.plot(x_axis, results['validation_0'][METODO_EVALUACION], label='Train')
    ax.plot(x_axis, results['validation_1'][METODO_EVALUACION], label='Test')
    ax.legend()
    pyplot.xlabel("Numero de epochs")
    pyplot.ylabel(METODO_EVALUACION)
    pyplot.title("Modelo: " + nombreModelo + " - Metodo de evaluacion: " + METODO_EVALUACION)
    path_img_metricas_modelo_ovft = dir_subgrupo_img + nombreModelo + "_" + METODO_EVALUACION + ".png"
    print("Pintando IMG de metricas del modelo overfitting (train vs test). Path: " + path_img_metricas_modelo_ovft)
    plt.savefig(path_img_metricas_modelo_ovft, bbox_inches='tight')
    plt.clf();        plt.cla();        plt.close();  # Limpiando dibujo
    #------------------------------------------------------------------------------


    # print("Se guarda el modelo " + nombreModelo + " en: " + pathModelo)
    if modeloEsGrid:
        s = pickle.dump(modelo.best_estimator_, open(pathModelo, 'wb'))
        out_grid_best_params = modelo.best_params_
        print("Modelo GRID tipo " + nombreModelo + " Los mejores parametros probados son: " + str(modelo.best_params_))

        if modoDebug and nombreModelo == "rf_grid":
            feature_imp = pd.Series(modelo.best_estimator_.feature_importances_, index=feature_names).sort_values(
                ascending=False)
            print("Importancia de las features en el modelo " + nombreModelo + " ha sido:")
            print(feature_imp.to_string())

            print("Generando dibujo de un árbol de decision (elegido al azar de los que haya)...")
            print(feature_names)
            print("Guardando dibujo DOT en: " + pathModelo + '.dot' + " Convertirlo ONLINE en: http://viz-js.com/")
            export_graphviz(modelo.best_estimator_.estimators_[1], out_file=pathModelo + '.dot',
                            feature_names=feature_names, class_names=list('TARGET'), rounded=True, proportion=False,
                            precision=2, filled=True)

            # Online Viewers:
            # http: // www.webgraphviz.com /
            # http: // sandbox.kidstrythisathome.com / erdos /
            # http: // viz - js.com /
            # Conversion local de DOT a PNG (en mi PC no consigo instalarlo):
            # call(['dot', '-Tpng', pathModelo + '.dot', '-o', pathModelo + '.png', '-Gdpi=600'])  # Convert to png

    else:
        s = pickle.dump(modelo, open(pathModelo, 'wb'))
    return modelo


def cargarModeloyUsarlo(dir_subgrupo_img, pathModelo, ds_test_f, ds_test_t, id_subgrupo, modoDebug):

    modelo_loaded = pickle.load(open(pathModelo, 'rb'))
    ds_test_t_pred = modelo_loaded.predict(
        ds_test_f)  # PREDICCION de los targets de TEST (los compararemos con los que tenemos)
    # area_bajo_roc = roc_auc_score(ds_test_t, ds_test_t_pred)
    # print(id_subgrupo + ' MODELO = ' + nombreModelo + " ROC_AUC score (test) = " + str(round(area_bajo_roc, 4)))
    # recall = recall_score(ds_test_t, ds_test_t_pred, average='binary', pos_label=1)
    # print(id_subgrupo + ' MODELO = ' + nombreModelo + ' Average RECALL score (test): {0:0.2f}'.format(recall))

    # precision_result = precision_score(ds_test_t, ds_test_t_pred)
    # print(id_subgrupo + ' MODELO = ' + nombreModelo + ' METRICA IMPORTANTE (test)--> precision score: {0:0.2f}'.format(precision_result) + " (num features = " + str(ds_test_f.shape[1]) + ")")

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
        plt.clf();
        plt.cla();
        plt.close();  # Limpiando dibujo

        print("Matriz de confusion...")
        path_dibujo = dir_subgrupo_img + nombreModelo + "_matriz_conf.png"
        disp = plot_confusion_matrix(modelo_loaded, ds_test_f, ds_test_t, cmap=plt.cm.Blues, normalize=None)
        disp.ax_.set_title(nombreModelo)
        print(disp.confusion_matrix)
        plt.savefig(path_dibujo, bbox_inches='tight')
        plt.clf();
        plt.cla();
        plt.close()  # Limpiando dibujo

    # return precision_result  # MAXIMIZAMOS la precision


################# MAIN ########################################
# GANADOR DEL SUBGRUPO (acumuladores)
ganador_nombreModelo = "NINGUNO"
ganador_metrica = 0
ganador_metrica_avg = 0
ganador_grid_mejores_parametros = []
pathListaColumnasCorreladasDrop = (dir_subgrupo + "columnas_correladas_drop" + ".txt").replace("futuro",
                                                                                               "pasado")  # lo guardamos siempre en el pasado

if (modoTiempo == "pasado" and pathCsvReducido.endswith('.csv') and os.path.isfile(pathCsvReducido) and os.stat(
        pathCsvReducido).st_size > 0):

    print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " Capa 6 - Modo PASADO")

    inputFeaturesyTarget = pd.read_csv(pathCsvReducido, index_col=0, sep='|')  # La columna 0 contiene el indice
    print("inputFeaturesyTarget: " + str(inputFeaturesyTarget.shape[0]) + " x " + str(inputFeaturesyTarget.shape[1]))

    print("BALANCEAR los casos positivos y negativos, haciendo downsampling de la clase mayoritaria...")
    print("URL: https://elitedatascience.com/imbalanced-classes")
    ift_mayoritaria = inputFeaturesyTarget[
        inputFeaturesyTarget.TARGET == False]  # En este caso los mayoritarios son los False
    ift_minoritaria = inputFeaturesyTarget[inputFeaturesyTarget.TARGET == True]
    print("ift_mayoritaria:" + str(ift_mayoritaria.shape[0]) + " x " + str(ift_mayoritaria.shape[1]))
    print("ift_minoritaria:" + str(ift_minoritaria.shape[0]) + " x " + str(ift_minoritaria.shape[1]))

    tasaDesbalanceoAntes = ift_mayoritaria.shape[0] / ift_minoritaria.shape[0]
    print("Tasa de desbalanceo entre clases (antes de balancear INICIO) = " + str(ift_mayoritaria.shape[0]) + "/" + str(ift_minoritaria.shape[0]) + " = " + str(tasaDesbalanceoAntes))
    num_muestras_minoria = ift_minoritaria.shape[0]

    casosInsuficientes = (num_muestras_minoria < umbralCasosSuficientesClasePositiva)
    if (casosInsuficientes):
        print("Numero de casos en clase minoritaria es INSUFICIENTE: "+str(num_muestras_minoria)+" (umbral="+str(umbralCasosSuficientesClasePositiva)+"). Así que abandonamos este dataset y seguimos")

    else:
        ift_juntas = pd.concat([ift_mayoritaria.reset_index(drop=True), ift_minoritaria.reset_index(drop=True)],
                               axis=0)  # Row bind
        indices_juntos = ift_mayoritaria.index.append(ift_minoritaria.index)  # Row bind
        ift_juntas.set_index(indices_juntos, inplace=True)
        print("Las clases juntas son:")
        print("ift_juntas:" + str(ift_juntas.shape[0]) + " x " + str(ift_juntas.shape[1]))

        ############ PANDAS PROFILING ###########
        if modoDebug:
            print("REDUCIDO - Profiling...")
            if len(ift_juntas) > 2000:
                prof = ProfileReport(ift_juntas.drop(columns=['TARGET']).sample(n=2000))
            else:
                prof = ProfileReport(ift_juntas.drop(columns=['TARGET']))

            prof.to_file(output_file=dir_subgrupo+"REDUCIDO_profiling.html")

        ###################### Matriz de correlaciones y quitar features correladas ###################
        print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " Matriz de correlaciones (PASADO):")
        matrizCorr = ift_juntas.corr().abs()
        # print(matrizCorr.to_string())
        upper = matrizCorr.where(np.triu(np.ones(matrizCorr.shape), k=1).astype(np.bool))
        # print(upper.to_string())
        print("Eliminamos las features muy correladas (umbral =" + str(umbralFeaturesCorrelacionadas) + "):")
        to_drop = [column for column in upper.columns if any(upper[column] > umbralFeaturesCorrelacionadas)]
        print(to_drop)
        print("Guardamos esa lista de features muy correladas en: " + pathListaColumnasCorreladasDrop)
        pickle.dump(to_drop, open(pathListaColumnasCorreladasDrop, 'wb'))
        ift_juntas.drop(to_drop, axis=1, inplace=True)
        #print(ift_juntas)
        print("Matriz de correlaciones corregida, habiendo quitado las muy correlacionadas (PASADO):")
        matrizCorr = ift_juntas.corr()
        #print(matrizCorr)
        print("matrizCorr:" + str(matrizCorr.shape[0]) + " x " + str(matrizCorr.shape[1]))
        ##################################################################
        columnasSeleccionadas = ift_juntas.columns
        print("Guardando las columnas seleccionadas en: ", pathFeaturesSeleccionadas)
        print(columnasSeleccionadas)
        columnasSeleccionadasStr = '|'.join(columnasSeleccionadas)
        featuresSeleccionadasFile = open(pathFeaturesSeleccionadas, "w")
        featuresSeleccionadasFile.write(columnasSeleccionadasStr)
        featuresSeleccionadasFile.close()
        ##################################################################

        print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " DIVIDIR EL DATASET DE ENTRADA EN 3 PARTES: TRAIN ("+str(fraccion_train)+"), TEST ("+str(fraccion_test)+"), VALIDACION ("+str(fraccion_valid)+")")
        ds_train, ds_test, ds_validacion = np.split(ift_juntas.sample(frac=1),
                                                    [int(fraccion_train * len(ift_juntas)),
                                                     int((fraccion_train+fraccion_test) * len(ift_juntas))])
        print("TRAIN = " + str(ds_train.shape[0]) + " x " + str(ds_train.shape[1]) + "  " + "TEST --> " + str(
            ds_test.shape[0]) + " x " + str(ds_test.shape[1]) + "  " + "VALIDACION --> " + str(
            ds_validacion.shape[0]) + " x " + str(ds_validacion.shape[1]))

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

        feature_names = ds_train.columns.drop('TARGET')

        ########################### SMOTE-ENN (Oversampling con SMOTE y undersampling con Edited Nearest Neighbours) ##################

        df_mayoritaria = ds_train_t[ds_train_t == False]  # En este caso los mayoritarios son los False
        df_minoritaria = ds_train_t[ds_train_t == True]
        print("df_mayoritaria:" + str(len(df_mayoritaria)))
        print("df_minoritaria:" + str(len(df_minoritaria)))
        tasaDesbalanceoAntes = len(df_mayoritaria) / len(df_minoritaria)
        print("Tasa de desbalanceo entre clases (antes de balancear con SMOTE) = " + str(len(df_mayoritaria)) + " / " + str(len(df_minoritaria)) + " = " + str(tasaDesbalanceoAntes))

        balancearConSmoteSoloTrain = (tasaDesbalanceoAntes > umbralNecesarioCompensarDesbalanceo)  # Condicion para decidir si hacer SMOTE
        if balancearConSmoteSoloTrain == True:
            print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " ---------------- RESAMPLING con SMOTE (porque supera umbral = " + str(umbralNecesarioCompensarDesbalanceo) + ") --------")
            print(
                "Resampling con SMOTE del vector de TRAINING (pero no a TEST ni a VALIDATION) según: " + "https://machinelearningmastery.com/combine-oversampling-and-undersampling-for-imbalanced-classification/")
            resample = SMOTEENN(sampling_strategy='minority',  # A que clase se hara el undersampling
                                random_state=0, smote=None, enn=None, n_jobs=-1)
            print("SMOTE antes (mayoritaria + minoritaria): %d" % ds_train_f.shape[0])
            ds_train_f, ds_train_t = resample.fit_resample(ds_train_f, ds_train_t)
            print("SMOTE después (mayoritaria + minoritaria): %d" % ds_train_f.shape[0])

            # print("Antes (mayoritaria + minoritaria): %d" % ds_train_f.shape[0])
            # # https://machinelearningmastery.com/undersampling-algorithms-for-imbalanced-classification/
            # undersample = NeighbourhoodCleaningRule(n_neighbors=3, threshold_cleaning=0.5)
            # ds_train_f, ds_train_t = undersample.fit_resample(ds_train_f, ds_train_t)
            # print("después (mayoritaria + minoritaria): %d" % ds_train_f.shape[0])
        ##################################################################

        ift_mayoritaria_entrada_modelos = ds_train_t[ds_train_t == False]  # En este caso los mayoritarios son los False
        ift_minoritaria_entrada_modelos = ds_train_t[ds_train_t == True]
        print("ift_mayoritaria_entrada_modelos:" + str(len(ift_mayoritaria_entrada_modelos)))
        print("ift_minoritaria_entrada_modelos:" + str(len(ift_minoritaria_entrada_modelos)))
        tasaDesbalanceoDespues = len(ift_mayoritaria_entrada_modelos) / len(ift_minoritaria_entrada_modelos)
        print("Tasa de desbalanceo entre clases (entrada a los modelos predictivos) = " + str(tasaDesbalanceoDespues))
        ##################################################################

        print("---------------- MODELOS con varias configuraciones (hiperparametros) --------")

        print("MODELOS: " + "https://scikit-learn.org/stable/supervised_learning.html")
        print(
            "EVALUACION de los modelos con: " + "https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics")
        print(
            "EVALUACION con curva precision-recall: " + "https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html")

        seed = 112  # Random seed

        #################### OPTIMIZACION DE PARAMETROS DE XGBOOST ###############################################################
        # Descomentar para obtener los parámetros con optimización Bayesiana
        # IMPORTANTE: se debe instalar el paquete de bayes en Conda: conda install -c conda-forge bayesian-optimization
        # Se imprimirán en el log, pero debo luego meterlos manualmente en el modelo
        # IMPORTANTE: DEBEN RELLENARSE 2 VALORES POR CADA ATRIBUTO DE PBOUND
        # https://ayguno.github.io/curious/portfolio/bayesian_optimization.html

        print("Inicio del optimizador de parametros de XGBOOST...")

        def xgboost_hyper_param(max_depth, learning_rate, n_estimators, reg_alpha, min_child_weight, colsample_bytree,
                                gamma):
            """Crea un modelo XGBOOST con los parametros indicados en la entrada. Aplica el numero de iteraciones de cross-validation indicado
                """
            clf = XGBClassifier(max_depth=int(max_depth), learning_rate=learning_rate, n_estimators=int(n_estimators),
                                reg_alpha=reg_alpha, min_child_weight=int(min_child_weight),
                                colsample_bytree=colsample_bytree, gamma=gamma,
                                nthread=-1, objective='binary:logistic', seed=seed)
            # "precision" tiene peores resultados que f1_weighted
            return np.mean(cross_val_score(clf, ds_train_f, ds_train_t, cv=cv_todos, scoring='f1_weighted'))

        # alpha is a parameter for the gaussian process
        # Note that this is itself a hyperparameter that can be optimized.
        gp_params = {"alpha": 1e-10}

        # Añadir carpeta dinámicamente: https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
        sys.path.append('/bayes_opt')
        from bayes_opt import BayesianOptimization

        pbounds = {
            'max_depth': (5, 20),
            'learning_rate': (0, 1),
            'n_estimators': (10, 100),
            'reg_alpha': (0, 1),
            'min_child_weight': (1, 20),
            'colsample_bytree': (0.1, 1),
            'gamma': (0, 10)
        }

        optimizer = BayesianOptimization(f=xgboost_hyper_param, pbounds=pbounds, random_state=1,
                                         verbose=10)
        optimizer.maximize(init_points=3, n_iter=10, acq='ucb', kappa=3, **gp_params)
        valoresOptimizados = optimizer.max
        print(valoresOptimizados)
        print("Fin del optimizador")
        ###################################################################################

        print("Inicio de XGBOOST")
        nombreModelo = "xgboost"
        pathModelo = dir_subgrupo + nombreModelo + ".modelo"
        # Parametros: https://xgboost.readthedocs.io/en/latest/parameter.html
        # Instalación en Conda: conda install -c anaconda py-xgboost
        # Instalación en Python básico: pip install xgboost

        # MODELO LUIS AUTOOPTIMIZADO PARA CADA SUBGRUPO
        max_depth = int(valoresOptimizados.get("params").get("max_depth"))
        learning_rate = valoresOptimizados.get("params").get("learning_rate")
        n_estimators = int(valoresOptimizados.get("params").get("n_estimators"))
        reg_alpha = valoresOptimizados.get("params").get("reg_alpha")
        min_child_weight = int(valoresOptimizados.get("params").get("min_child_weight"))
        colsample_bytree = valoresOptimizados.get("params").get("colsample_bytree")
        gamma = valoresOptimizados.get("params").get("gamma")
        nthread = -1
        objective = 'binary:logistic'
        seed = seed

        modelo = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators,
                               reg_alpha=reg_alpha, min_child_weight=min_child_weight,
                               colsample_bytree=colsample_bytree, gamma=gamma,
                               nthread=nthread, objective=objective, seed=seed)
        modelo_xgboost = ejecutarModeloyGuardarlo(nombreModelo, modelo, pathModelo, ds_train_f,
                                                  ds_train_t, ds_test_f, ds_test_t, feature_names, False, modoDebug,
                                                  dir_subgrupo_img)
        cargarModeloyUsarlo(dir_subgrupo_img, pathModelo, ds_test_f, ds_test_t, id_subgrupo, modoDebug)

        test_t_predicho = modelo_xgboost.predict(ds_test_f);
        validac_t_predicho = modelo_xgboost.predict(ds_validac_f)
        precision_test = precision_score(ds_test_t, test_t_predicho);
        precision_avg_test = average_precision_score(ds_test_t, test_t_predicho)
        precision_validation = precision_score(ds_validac_t, validac_t_predicho);
        precision_avg_validation = average_precision_score(ds_validac_t, validac_t_predicho)
        precision_media = (precision_test + precision_validation) / 2
        precision_avg_media = (precision_avg_test + precision_avg_validation) / 2
        print(id_subgrupo + " " + nombreModelo + " -> Precision = " + '{0:0.2f}'.format(
            precision_media) + " (average precision = " + '{0:0.2f}'.format(precision_avg_media) + ")")
        if precision_media > ganador_metrica:
            ganador_metrica = precision_media
            ganador_metrica_avg = precision_avg_media
            ganador_nombreModelo = nombreModelo
            ganador_grid_mejores_parametros = []  # Este modelo no tiene esta variable
        print("Fin de XGBOOST")


        ######################################################################################################################
        ######################################################################################################################

        print("********* GANADOR de subgrupo *************")
        print("PASADO -> " + id_subgrupo + " (num features = " + str(
            ds_train_f.shape[1]) + ")" + " -> Modelo ganador = " + ganador_nombreModelo + " --> METRICA = " + str(
            round(ganador_metrica, 4)) + " (avg_precision = " + str(round(ganador_metrica_avg, 4)) + ") ")
        print("Hiperparametros:")
        print(ganador_grid_mejores_parametros)
        pathModeloGanadorDeSubgrupoOrigen = dir_subgrupo + ganador_nombreModelo + ".modelo"
        pathModeloGanadorDeSubgrupoDestino = pathModeloGanadorDeSubgrupoOrigen + "_ganador"
        copyfile(pathModeloGanadorDeSubgrupoOrigen, pathModeloGanadorDeSubgrupoDestino)
        print("Modelo ganador guardado en: " + pathModeloGanadorDeSubgrupoDestino)

        ######################################################################################################################
        ######################################################################################################################


elif (modoTiempo == "futuro" and pathCsvReducido.endswith('.csv') and os.path.isfile(pathCsvReducido) and os.stat(
        pathCsvReducido).st_size > 0):

    print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " Capa 6 - Modo futuro")
    inputFeaturesyTarget = pd.read_csv(pathCsvReducido, index_col=0, sep='|')
    print("inputFeaturesyTarget: " + str(inputFeaturesyTarget.shape[0]) + " x " + str(inputFeaturesyTarget.shape[1]))

    print("Si las hay, eliminamos las features muy correladas (umbral =" + str(
        umbralFeaturesCorrelacionadas) + ") aprendido en el PASADO..")
    if os.path.exists(pathListaColumnasCorreladasDrop):
        columnasCorreladas = pickle.load(open(pathListaColumnasCorreladasDrop, 'rb'))
        inputFeaturesyTarget.drop(columnasCorreladas, axis=1, inplace=True)
        print(columnasCorreladas)

    print("Matriz de correlaciones corregida (FUTURO):")
    matrizCorr = inputFeaturesyTarget.corr()
    print(matrizCorr)

    print(
        "La columna TARGET que haya en el CSV de entrada no la queremos (es un NULL o False, por defecto), porque la vamos a PREDECIR...")
    inputFeatures = inputFeaturesyTarget.drop('TARGET', axis=1)
    print(inputFeatures.head())
    print("inputFeatures: " + str(inputFeatures.shape[0]) + " x " + str(inputFeatures.shape[1]))

    print(
        "MISSING VALUES (FILAS) - Borramos las FILAS que tengan 1 o mas valores NaN porque son huecos que no deberían estar...")
    inputFeatures_sinnulos = inputFeatures.dropna(axis=0, how='any')  # Borrar FILA si ALGUNO sus valores tienen NaN

    dir_modelo_predictor_ganador = dir_subgrupo.replace("futuro",
                                                        "pasado")  # Siempre cojo el modelo entrenado en el pasado
    path_modelo_predictor_ganador = ""
    for file in os.listdir(dir_modelo_predictor_ganador):
        if file.endswith("ganador"):
            path_modelo_predictor_ganador = os.path.join(dir_modelo_predictor_ganador, file)

    print("Cargar modelo PREDICTOR ganador (de la carpeta del pasado, SI EXISTE): " + path_modelo_predictor_ganador)
    if os.path.isfile(path_modelo_predictor_ganador):

        modelo_predictor_ganador = pickle.load(open(path_modelo_predictor_ganador, 'rb'))

        print("Predecir:")
        cols_when_model_builds = modelo_predictor_ganador.get_booster().feature_names
        # El modelo XGB guarda como nombres de los features: fx, donde x=0, 1, 2... Si es así, se cambiará el
        # nombre de los features de entrada al predictor
        i = 0
        inputFeatures_sinnulosNuevo = pd.DataFrame()
        if 'f0' in cols_when_model_builds:
            for nombreFeature in inputFeatures_sinnulos.columns:
                nuevoNombre = 'f' + str(i)
                print("Antes de predecir se sustituye " + nombreFeature + " por " + nuevoNombre)
                inputFeatures_sinnulosNuevo[nuevoNombre] = inputFeatures_sinnulos[nombreFeature]
                i += 1
            inputFeatures_sinnulos = inputFeatures_sinnulosNuevo

        targets_predichos = modelo_predictor_ganador.predict(inputFeatures_sinnulos)
        num_targets_predichos = len(targets_predichos)
        print("Numero de targets_predichos: " + str(num_targets_predichos) + " con numero de TRUEs = " + str(
            np.sum(targets_predichos, where=["True"])))
        # print("El array de targets contiene:")
        # print(targets_predichos)

        # probabilities
        probs = pd.DataFrame(data=modelo_predictor_ganador.predict_proba(inputFeatures_sinnulos),
                             index=inputFeatures_sinnulos.index)

        # UMBRAL MENOS PROBABLES CON TARGET=1. Cuando el target es 1, se guarda su probabilidad
        # print("El DF llamado probs contiene las probabilidades de predecir un 0 o un 1:")
        # print(probs)

        probabilidadesEnTargetUnoPeq = probs.iloc[:, 1]  # Cogemos solo la segunda columna: prob de que sea target=1
        probabilidadesEnTargetUnoPeq2 = probabilidadesEnTargetUnoPeq.apply(
            lambda x: x if (x >= umbralProbTargetTrue) else np.nan) # Cogemos solo las filas cuya prob_1 > umbral
        probabilidadesEnTargetUnoPeq3 = probabilidadesEnTargetUnoPeq2[
            np.isnan(probabilidadesEnTargetUnoPeq2[:]) == False]  # Cogemos todos los no nulos (NAN)
        # print("El DF llamado probabilidadesEnTargetUnoPeq3 contiene las probabilidades de los UNO con prob mayor que umbral ("+str(umbralProbTargetTrue)+"):")
        # print(probabilidadesEnTargetUnoPeq3)

        probabilidadesEnTargetUnoPeq4 = probabilidadesEnTargetUnoPeq3.sort_values(ascending=False)  # descendente
        print("El DF llamado probabilidadesEnTargetUnoPeq4 contiene los indices y probabilidades, tras aplicar umbral INFERIOR: " + str(umbralProbTargetTrue) + ". Son:")
        print(probabilidadesEnTargetUnoPeq4)

        numfilasSeleccionadas = int(granProbTargetUno * probabilidadesEnTargetUnoPeq4.shape[
            0] / 100)  # Como están ordenadas en descendente, cojo estas NUM primeras filas
        print("numfilasSeleccionadas: " + str(numfilasSeleccionadas))
        targets_predichosCorregidos_probs = probabilidadesEnTargetUnoPeq4[0:numfilasSeleccionadas]
        targets_predichosCorregidos = targets_predichosCorregidos_probs.apply(lambda x: 1)
        print(
            "El DF llamado targets_predichosCorregidos contiene los indices y probabilidades, tras aplicar umbral SUPERIOR: top " + str(
                granProbTargetUno) + " % de muestras. Son:")
        print(targets_predichosCorregidos)

        print("Guardando targets PREDICHOS en: " + pathCsvPredichos)
        df_predichos = targets_predichosCorregidos.to_frame()
        df_predichos.columns = ['TARGET_PREDICHO']
        df_predichos.to_csv(pathCsvPredichos, index=False, sep='|')  # Capa 6 - Salida (para el validador, sin indice)

        df_predichos_probs = targets_predichosCorregidos_probs.to_frame()
        df_predichos_probs.columns = ['TARGET_PREDICHO_PROB']
        df_predichos_probs.to_csv(pathCsvPredichos + "_humano", index=True, sep='|')  # Capa 6 - Salida (para el humano)

        ############### RECONSTRUCCION DEL CSV FINAL IMPORTANTE, viendo los ficheros de indices #################
        print(
            "Partiendo de COMPLETO.csv llevamos la cuenta de los indices pasando por REDUCIDO.csv y por TARGETS_PREDICHOS.csv para generar el CSV final...")
        df_completo = pd.read_csv(pathCsvCompleto, sep='|')  # Capa 5 - Entrada

        print("df_completo: " + str(df_completo.shape[0]) + " x " + str(df_completo.shape[1]))
        print("df_predichos: " + str(df_predichos.shape[0]) + " x " + str(df_predichos.shape[1]))
        print("df_predichos_probs: " + str(df_predichos_probs.shape[0]) + " x " + str(df_predichos_probs.shape[1]))

        print("Juntar COMPLETO con TARGETS PREDICHOS... ")
        df_juntos_1 = pd.concat([df_completo, df_predichos], axis=1)
        df_juntos_2 = pd.concat([df_juntos_1, df_predichos_probs], axis=1)

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
print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") +" ------------ FIN de capa 6----------------")
