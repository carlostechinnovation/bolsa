import sys
import pandas as pd
import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from numpy import mean
from scipy.stats import stats, chi2
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.fixes import loguniform
from pathlib import Path
from sklearn import svm
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, recall_score, make_scorer, \
    precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from shutil import copyfile
import os.path
from imblearn.over_sampling import SMOTE
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.tree import export_graphviz
from subprocess import call
from xgboost import XGBClassifier
# from bolsa.bayes_opt import BayesianOptimization

np.random.seed(12345)

print("\n **** CAPA 6 - Crear almacenar y evaluar varios modelos (para cada subgrupo) ****")
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
umbralFeaturesCorrelacionadas = 0.9
cv_todos = 10

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


def ejecutarModeloyGuardarlo(nombreModelo, modelo, pathModelo, ds_train_f, ds_train_t, feature_names, modeloEsGrid,
                             modoDebug):
    print("Ejecutando " + nombreModelo + " ...")
    out_grid_best_params = []

    modelo.fit(ds_train_f, ds_train_t)  # ENTRENAMIENTO (TRAIN)

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
    return out_grid_best_params


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

    inputFeaturesyTarget = pd.read_csv(pathCsvReducido, index_col=0, sep='|')  # La columna 0 contiene el indice
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
        ift_juntas = pd.concat([ift_mayoritaria.reset_index(drop=True), ift_minoritaria.reset_index(drop=True)],
                               axis=0)  # Row bind
        indices_juntos = ift_mayoritaria.index.append(ift_minoritaria.index)  # Row bind
        ift_juntas.set_index(indices_juntos, inplace=True)
        print("Las clases juntas son:")
        print("ift_juntas:" + str(ift_juntas.shape[0]) + " x " + str(ift_juntas.shape[1]))

        ###################### Matriz de correlaciones y quitar features correladas ###################
        print("Matriz de correlaciones (PASADO):")
        matrizCorr = ift_juntas.corr().abs()
        print(matrizCorr.to_string())
        upper = matrizCorr.where(np.triu(np.ones(matrizCorr.shape), k=1).astype(np.bool))
        print(upper.to_string())
        print("Eliminamos las features muy correladas (umbral =" + str(umbralFeaturesCorrelacionadas) + "):")
        to_drop = [column for column in upper.columns if any(upper[column] > umbralFeaturesCorrelacionadas)]
        print(to_drop)
        print("Guardamos esa lista de features muy correladas en: " + pathListaColumnasCorreladasDrop)
        pickle.dump(to_drop, open(pathListaColumnasCorreladasDrop, 'wb'))
        ift_juntas.drop(to_drop, axis=1, inplace=True)
        #print(ift_juntas)
        print("Matriz de correlaciones corregida, habiendo quitado las muy correlacionadas (PASADO):")
        matrizCorr = ift_juntas.corr()
        print(matrizCorr)
        ##################################################################
        columnasSeleccionadas = ift_juntas.columns
        print("Guardando las columnas seleccionadas en: ", pathFeaturesSeleccionadas)
        print(columnasSeleccionadas)
        columnasSeleccionadasStr = '|'.join(columnasSeleccionadas)
        featuresSeleccionadasFile = open(pathFeaturesSeleccionadas, "w")
        featuresSeleccionadasFile.write(columnasSeleccionadasStr)
        featuresSeleccionadasFile.close()
        ##################################################################

        print("DIVIDIR EL DATASET DE ENTRADA EN 3 PARTES: TRAIN (50%), TEST (25%), VALIDACION (25%)...")
        ds_train, ds_test, ds_validacion = np.split(ift_juntas.sample(frac=1),
                                                    [int(0.5 * len(ift_juntas)), int(0.75 * len(ift_juntas))])
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

        # ################### Según un caso de Kaggle, se transforma alguna columna para que se vean bien
        # ################### las diferencias: https://www.kaggle.com/jesucristo/1-house-prices-solution-top-1
        # ################### Se aplica sólo en el training, con logaritmos
        # columnasAEstirar=[
        #     'volumen', 'low', 'close', 'PENDIENTE_2M_SMA_4_PRECIO', 'RATIO_SMA_4_PRECIO', 'RATIO_U_MINRELATIVO_4_PRECIO',
        #  'PENDIENTE_2M_SMA_7_PRECIO', 'RATIO_U_MAXRELATIVO_7_PRECIO', 'SKEWNESS_7_PRECIO',
        #  'RATIO_U_MAXRELATIVO_7_VOLUMEN', 'SKEWNESS_7_VOLUMEN', 'PENDIENTE_SMA_20_PRECIO', 'PENDIENTE_1M_SMA_20_PRECIO',
        #  'RATIO_SMA_20_PRECIO', 'RATIO_MAXRELATIVO_20_PRECIO', 'RATIO_MINRELATIVO_20_PRECIO',
        #  'RATIO_U_MINRELATIVO_20_PRECIO', 'CURTOSIS_20_PRECIO', 'RATIO_U_MAXRELATIVO_20_VOLUMEN', 'SKEWNESS_20_VOLUMEN',
        #  'PENDIENTE_SMA_50_PRECIO', 'PENDIENTE_1M_SMA_50_PRECIO', 'RATIO_SMA_50_PRECIO', 'RATIO_MAXRELATIVO_50_PRECIO',
        #  'RATIO_MINRELATIVO_50_PRECIO', 'RATIO_U_SMA_50_PRECIO', 'RATIO_U_MAXRELATIVO_50_PRECIO',
        #  'RATIO_U_MINRELATIVO_50_PRECIO', 'CURTOSIS_50_PRECIO', 'RATIO_MAXRELATIVO_50_VOLUMEN'
        # ]
        # for nombreColumna in columnasAEstirar:
        #     if nombreColumna in ds_train.columns:
        #         min=np.min(ds_train[nombreColumna])
        #         max = np.max(ds_train[nombreColumna])
        #         mean = np.mean(ds_train[nombreColumna])
        #         median = np.median(ds_train[nombreColumna])
        #         print("\n feature: "+nombreColumna+"\t min: "+str(min)+"\t max "+str(max)+
        #               "\t mean: "+str(mean)+"\t median: "+str(median))
        #         ds_train[nombreColumna] = np.log1p(ds_train[nombreColumna])
        #
        # ###############################################################################################

        # # ################### Se aplica al training, desplazar para evitar valores negativos, y
        # # achatar los rangos muy grandes/estirar los pequeños
        # for nombreColumna in ds_train.columns:
        #     min = np.min(ds_train[nombreColumna])
        #     max = np.max(ds_train[nombreColumna])
        #     mean = np.mean(ds_train[nombreColumna])
        #     median = np.median(ds_train[nombreColumna])
        #     print("\n feature: " + nombreColumna + "\t min: " + str(min) + "\t max " + str(max) +
        #           "\t mean: " + str(mean) + "\t median: " + str(median))
        #     # Se evitan números negativos, desplazando la feature el absoluto de la cantidad mínima
        #     if min < 0:
        #         ds_train[nombreColumna] = ds_train[nombreColumna] - min
        #     # Si, una vez desplazado, la diferencia entre mínimo y máximo es mayor que 100, se aplica logaritmo
        #     # para achatar las diferencias
        #     min = np.min(ds_train[nombreColumna])
        #     max = np.max(ds_train[nombreColumna])
        #     if min == 0 or (max / min > 100):
        #         ds_train[nombreColumna] = np.log1p(ds_train[nombreColumna])
        #     else:
        #         ds_train[nombreColumna] = np.power(ds_train[nombreColumna], 2)
        # ################################################################################################################

        ########################### SMOTE (Over and undersampling on the train data) ##################
        if (balancearConSmoteSoloTrain == True):
            print("---------------- RESAMPLING con SMOTE --------")
            print(
                "Resampling con SMOTE del vector de TRAINING (pero no a TEST ni a VALIDATION) según: " + "https://machinelearningmastery.com/combine-oversampling-and-undersampling-for-imbalanced-classification/")
            resample = SMOTEENN()
            print("SMOTE antes (mayoritaria + minoritaria): %d" % ds_train_f.shape[0])
            ds_train_f, ds_train_t = resample.fit_sample(ds_train_f, ds_train_t)
            print("SMOTE después (mayoritaria + minoritaria): %d" % ds_train_f.shape[0])
            ##################################################################

            print("---------------- MODELOS con varias configuraciones (hiperparametros) --------")

            print("MODELOS: " + "https://scikit-learn.org/stable/supervised_learning.html")
            print(
                "EVALUACION de los modelos con: " + "https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics")
            print(
                "EVALUACION con curva precision-recall: " + "https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html")

            # Parametros por defecto de los modelos que usan árboles de decisión
            ARBOLES_n_estimators = 80
            ARBOLES_max_depth = 11
            ARBOLES_min_samples_leaf = 20
            ARBOLES_max_features = "auto"
            ARBOLES_min_samples_split = 3
            ARBOLES_max_leaf_nodes = None
            ARBOLES_min_impurity_decrease = 0.001

            # nombreModelo = "gradient_boosting"
            # pathModelo = dir_subgrupo + nombreModelo + ".modelo"
            # modelo = GradientBoostingClassifier(n_estimators=3000, learning_rate=0.05,
            #                           max_depth=4, max_features='sqrt',
            #                           min_samples_leaf=15, min_samples_split=10, random_state=42)
            # modelo_grid_mejores_parametros = ejecutarModeloyGuardarlo(nombreModelo, modelo, pathModelo, ds_train_f,
            #                                                           ds_train_t, feature_names, False, modoDebug)
            # modelo_metrica = cargarModeloyUsarlo(dir_subgrupo_img, pathModelo, ds_test_f, ds_test_t, id_subgrupo,
            #                                      modoDebug)
            # print(type(modelo_metrica))
            # if modelo_metrica > ganador_metrica:
            #     ganador_metrica = modelo_metrica
            #     ganador_metrica_avg = modelo_metrica
            #     ganador_nombreModelo = nombreModelo
            #     ganador_grid_mejores_parametros = modelo_grid_mejores_parametros

            # # Descomentar para obtener los parámetros con optimización Bayesiana
            # # Se imprimirán en el log, pero debo luego meterlos manualmente en el modelo
            # pbounds = {
            #     'learning_rate': (0.1, 1.0),
            #     'n_estimators': (100, 1000),
            #     'max_depth': (3, 10),
            #     'subsample': (0.8, 1.0),  # Change for big datasets
            #     'colsample_bytree': (0.2, 1.0),  # Change for datasets with lots of features
            #     'gamma': (0, 15)}
            # def xgboost_hyper_param(learning_rate,
            #                         n_estimators,
            #                         max_depth,
            #                         subsample,
            #                         colsample_bytree,
            #                         gamma):
            #     max_depth = int(max_depth)
            #     n_estimators = int(n_estimators)
            #     clf = XGBClassifier(
            #         max_depth=max_depth,
            #         learning_rate=learning_rate,
            #         n_estimators=n_estimators,
            #         gamma=gamma)
            #     return np.mean(cross_val_score(clf, ds_train_f, ds_train_t, cv=5, scoring='precision'))
            # optimizer = BayesianOptimization(
            #     f=xgboost_hyper_param,
            #     pbounds=pbounds,
            #     random_state=1,
            # )
            # optimizer.maximize(init_points=10, n_iter=50)
            # valoresOptimizados=optimizer.max
            # print(valoresOptimizados)
            # print("Inicio de XGBOOST")
            nombreModelo = "xgboost"
            pathModelo = dir_subgrupo + nombreModelo + ".modelo"
            modelo = XGBClassifier(learning_rate=0.1174, n_estimators=291,
                                   max_depth=8,
                                   gamma=1.95, subsample=0.8531,
                                   colsample_bytree=0.4302)
            modelo_grid_mejores_parametros = ejecutarModeloyGuardarlo(nombreModelo, modelo, pathModelo, ds_train_f,
                                                                      ds_train_t, feature_names, False, modoDebug)
            cargarModeloyUsarlo(dir_subgrupo_img, pathModelo, ds_test_f, ds_test_t, id_subgrupo, modoDebug)

            test_t_predicho = modelo.predict(ds_test_f);
            validac_t_predicho = modelo.predict(ds_validac_f)
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
                ganador_grid_mejores_parametros = modelo_grid_mejores_parametros
            print("Fin de XGBOOST")

            # #============================================================================
            # nombreModelo = "extra_trees"
            # pathModelo = dir_subgrupo + nombreModelo + ".modelo"
            # modelo = ExtraTreesClassifier(n_estimators=ARBOLES_n_estimators, max_depth=ARBOLES_max_depth, min_samples_leaf=ARBOLES_min_samples_leaf,
            #                               max_features=ARBOLES_max_features, min_samples_split=ARBOLES_min_samples_split,
            #                               max_leaf_nodes=ARBOLES_max_leaf_nodes, min_impurity_decrease=ARBOLES_min_impurity_decrease,
            #                               criterion="gini", min_weight_fraction_leaf=0., min_impurity_split=None, bootstrap=False, oob_score=False,
            #                               n_jobs=None, random_state=1, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
            # modelo_grid_mejores_parametros = ejecutarModeloyGuardarlo(nombreModelo, modelo, pathModelo, ds_train_f, ds_train_t, feature_names, False, modoDebug)
            # cargarModeloyUsarlo(dir_subgrupo_img, pathModelo, ds_test_f, ds_test_t, id_subgrupo, modoDebug)
            #
            # test_t_predicho = modelo.predict(ds_test_f); validac_t_predicho = modelo.predict(ds_validac_f)
            # precision_test = precision_score(ds_test_t, test_t_predicho); precision_avg_test = average_precision_score(ds_test_t, test_t_predicho)
            # precision_validation = precision_score(ds_validac_t, validac_t_predicho); precision_avg_validation = average_precision_score(ds_validac_t, validac_t_predicho)
            # precision_media = (precision_test + precision_validation) / 2
            # precision_avg_media = (precision_avg_test + precision_avg_validation) / 2
            # print(id_subgrupo + " " + nombreModelo + " -> Precision = " + '{0:0.2f}'.format(precision_media) + " (average precision = " + '{0:0.2f}'.format(precision_avg_media) + ")")
            # if precision_media > ganador_metrica:
            #     ganador_metrica = precision_media
            #     ganador_metrica_avg = precision_avg_media
            #     ganador_nombreModelo = nombreModelo
            #     ganador_grid_mejores_parametros = modelo_grid_mejores_parametros
            #
            #
            # # ============================================================================
            # nombreModelo = "nn"  # MultiLayer Perceptron
            # pathModelo = dir_subgrupo + nombreModelo + ".modelo"
            # modelo = MLPClassifier(hidden_layer_sizes=(20, 5), activation="tanh", solver='lbfgs', alpha=0.0001, batch_size='auto', learning_rate="constant",
            #                        learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=1, tol=1e-4,
            #                        verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False,
            #                        validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8, n_iter_no_change=10, max_fun=15000)
            #
            # modelo_grid_mejores_parametros = ejecutarModeloyGuardarlo(nombreModelo, modelo, pathModelo, ds_train_f, ds_train_t, feature_names, False, modoDebug)
            # cargarModeloyUsarlo(dir_subgrupo_img, pathModelo, ds_test_f, ds_test_t, id_subgrupo, modoDebug)
            #
            # test_t_predicho = modelo.predict(ds_test_f);
            # validac_t_predicho = modelo.predict(ds_validac_f)
            # precision_test = precision_score(ds_test_t, test_t_predicho);
            # precision_avg_test = average_precision_score(ds_test_t, test_t_predicho)
            # precision_validation = precision_score(ds_validac_t, validac_t_predicho);
            # precision_avg_validation = average_precision_score(ds_validac_t, validac_t_predicho)
            # precision_media = (precision_test + precision_validation) / 2
            # precision_avg_media = (precision_avg_test + precision_avg_validation) / 2
            # print(id_subgrupo + " " + nombreModelo + " -> Precision = " + '{0:0.2f}'.format(
            #     precision_media) + " (average precision = " + '{0:0.2f}'.format(precision_avg_media) + ")")
            # if precision_media > ganador_metrica:
            #     ganador_metrica = precision_media
            #     ganador_metrica_avg = precision_avg_media
            #     ganador_nombreModelo = nombreModelo
            #     ganador_grid_mejores_parametros = modelo_grid_mejores_parametros
            #
            #
            # # ################### Muchos modelos, usando GRID de parametros ###################################################
            # print("HYPERPARAMETROS - URL: https://scikit-learn.org/stable/modules/grid_search.html")
            #
            # # ============================================================================
            # # nombreModelo = "svc_grid"
            # # pathModelo = dir_subgrupo + nombreModelo + ".modelo"
            # # modelo_base = svm.SVC()
            # # hiperparametros = [{'C': [10, 50], 'gamma':[10, 30], 'kernel':['rbf']}]
            # # modelos_grid = GridSearchCV(modelo_base, hiperparametros, scoring='precision', n_jobs=-1, refit=True, cv=cv_todos, pre_dispatch='2*n_jobs', return_train_score=False)
            # # modelo_grid_mejores_parametros = ejecutarModeloyGuardarlo(nombreModelo, modelos_grid, pathModelo, ds_train_f, ds_train_t, feature_names, True, modoDebug)
            # # cargarModeloyUsarlo(dir_subgrupo_img, pathModelo, ds_test_f, ds_test_t, id_subgrupo, modoDebug)
            # #
            # # test_t_predicho = modelo.predict(ds_test_f);
            # # validac_t_predicho = modelo.predict(ds_validac_f)
            # # precision_test = precision_score(ds_test_t, test_t_predicho);
            # # precision_avg_test = average_precision_score(ds_test_t, test_t_predicho)
            # # precision_validation = precision_score(ds_validac_t, validac_t_predicho);
            # # precision_avg_validation = average_precision_score(ds_validac_t, validac_t_predicho)
            # # precision_media = (precision_test + precision_validation) / 2
            # # precision_avg_media = (precision_avg_test + precision_avg_validation) / 2
            # # print(id_subgrupo + " " + nombreModelo + " -> Precision = " + '{0:0.2f}'.format(
            # #     precision_media) + " (average precision = " + '{0:0.2f}'.format(precision_avg_media) + ")")
            # # if precision_media > ganador_metrica:
            # #     ganador_metrica = precision_media
            # #     ganador_metrica_avg = precision_avg_media
            # #     ganador_nombreModelo = nombreModelo
            # #     ganador_grid_mejores_parametros = modelo_grid_mejores_parametros
            #
            # # ============================================================================
            # nombreModelo = "nn_grid"
            # pathModelo = dir_subgrupo + nombreModelo + ".modelo"
            # modelo_base = MLPClassifier(hidden_layer_sizes=(5, 2), activation="relu", solver='lbfgs', alpha=0.0001, batch_size='auto', learning_rate="constant",
            #                        learning_rate_init=0.01, power_t=0.5, max_iter=200, shuffle=True, random_state=1, tol=1e-2,
            #                        verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False,
            #                        validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8, n_iter_no_change=10,
            #                        max_fun=15000)
            # hiperparametros = {'hidden_layer_sizes': [(5, 2), (20, 5), (50, 20)], 'solver': ['lbfgs'], 'activation': ['identity', 'logistic', 'tanh', 'relu']}
            # modelos_grid = GridSearchCV(modelo_base, hiperparametros, scoring='precision', n_jobs=-1, refit=True, cv=cv_todos,pre_dispatch='2*n_jobs', return_train_score=False)
            # modelo_grid_mejores_parametros = ejecutarModeloyGuardarlo(nombreModelo, modelos_grid, pathModelo,ds_train_f, ds_train_t, feature_names, True, modoDebug)
            # cargarModeloyUsarlo(dir_subgrupo_img, pathModelo, ds_test_f, ds_test_t, id_subgrupo, modoDebug)
            # test_t_predicho = modelo.predict(ds_test_f);
            # validac_t_predicho = modelo.predict(ds_validac_f)
            # precision_test = precision_score(ds_test_t, test_t_predicho);
            # precision_avg_test = average_precision_score(ds_test_t, test_t_predicho)
            # precision_validation = precision_score(ds_validac_t, validac_t_predicho);
            # precision_avg_validation = average_precision_score(ds_validac_t, validac_t_predicho)
            # precision_media = (precision_test + precision_validation) / 2
            # precision_avg_media = (precision_avg_test + precision_avg_validation) / 2
            # print(id_subgrupo + " " + nombreModelo + " -> Precision = " + '{0:0.2f}'.format(
            #     precision_media) + " (average precision = " + '{0:0.2f}'.format(precision_avg_media) + ")")
            # if precision_media > ganador_metrica:
            #     ganador_metrica = precision_media
            #     ganador_metrica_avg = precision_avg_media
            #     ganador_nombreModelo = nombreModelo
            #     ganador_grid_mejores_parametros = modelo_grid_mejores_parametros
            #
            #
            # # ============================================================================
            # nombreModelo = "logreg_grid"
            # pathModelo = dir_subgrupo + nombreModelo + ".modelo"
            # modelo_base = LogisticRegression(penalty='l2', dual=False, tol=1e-4, C=1.0,
            #      fit_intercept=True, intercept_scaling=1, class_weight=None,
            #      random_state=None, solver='lbfgs', max_iter=100,
            #      multi_class='auto', verbose=0, warm_start=False, n_jobs=-1,
            #      l1_ratio=None)
            # hiperparametros = dict(C=np.logspace(0, 2, num=10, base=10), penalty=['l2'])
            # modelos_grid = GridSearchCV(modelo_base, hiperparametros, scoring='precision', n_jobs=-1, refit=True, cv=cv_todos, pre_dispatch='2*n_jobs', return_train_score=False)
            # modelo_grid_mejores_parametros = ejecutarModeloyGuardarlo(nombreModelo, modelos_grid, pathModelo, ds_train_f, ds_train_t, feature_names, True, modoDebug)
            # cargarModeloyUsarlo(dir_subgrupo_img, pathModelo, ds_test_f, ds_test_t, id_subgrupo, modoDebug)
            #
            # logreg_coef = modelos_grid.best_estimator_.coef_
            # logreg_coef = pd.DataFrame(data=logreg_coef, columns=feature_names)
            # print("Pesos en regresion logistica:"); print(logreg_coef.to_string())
            #
            # test_t_predicho = modelo.predict(ds_test_f);
            # validac_t_predicho = modelo.predict(ds_validac_f)
            # precision_test = precision_score(ds_test_t, test_t_predicho);
            # precision_avg_test = average_precision_score(ds_test_t, test_t_predicho)
            # precision_validation = precision_score(ds_validac_t, validac_t_predicho);
            # precision_avg_validation = average_precision_score(ds_validac_t, validac_t_predicho)
            # precision_media = (precision_test + precision_validation) / 2
            # precision_avg_media = (precision_avg_test + precision_avg_validation) / 2
            # print(id_subgrupo + " " + nombreModelo + " -> Precision = " + '{0:0.2f}'.format(
            #     precision_media) + " (average precision = " + '{0:0.2f}'.format(precision_avg_media) + ")")
            # if precision_media > ganador_metrica:
            #     ganador_metrica = precision_media
            #     ganador_metrica_avg = precision_avg_media
            #     ganador_nombreModelo = nombreModelo
            #     ganador_grid_mejores_parametros = modelo_grid_mejores_parametros
            #
            #
            # # ============================================================================
            # nombreModelo = "rf_grid"
            # pathModelo = dir_subgrupo + nombreModelo + ".modelo"
            # modelo_base = RandomForestClassifier(n_estimators=ARBOLES_n_estimators, max_depth=ARBOLES_max_depth, min_samples_leaf=ARBOLES_min_samples_leaf,
            #                                      max_features=ARBOLES_max_features, min_samples_split=ARBOLES_min_samples_split,
            #                                      max_leaf_nodes=ARBOLES_max_leaf_nodes, min_impurity_decrease=ARBOLES_min_impurity_decrease, random_state=1)
            # hiperparametros = {'min_impurity_decrease': [0.001, 0.00001], 'max_depth': [9, 11, 13]}
            # modelos_grid = GridSearchCV(modelo_base, hiperparametros, scoring='precision', n_jobs=-1, refit=True, return_train_score=False, cv=cv_todos)
            # modelo_grid_mejores_parametros = ejecutarModeloyGuardarlo(nombreModelo, modelos_grid, pathModelo, ds_train_f, ds_train_t, feature_names, True, modoDebug)
            # cargarModeloyUsarlo(dir_subgrupo_img, pathModelo, ds_test_f, ds_test_t, id_subgrupo, modoDebug)
            #
            # test_t_predicho = modelo.predict(ds_test_f);
            # validac_t_predicho = modelo.predict(ds_validac_f)
            # precision_test = precision_score(ds_test_t, test_t_predicho);
            # precision_avg_test = average_precision_score(ds_test_t, test_t_predicho)
            # precision_validation = precision_score(ds_validac_t, validac_t_predicho);
            # precision_avg_validation = average_precision_score(ds_validac_t, validac_t_predicho)
            # precision_media = (precision_test + precision_validation) / 2
            # precision_avg_media = (precision_avg_test + precision_avg_validation) / 2
            # print(id_subgrupo + " " + nombreModelo + " -> Precision = " + '{0:0.2f}'.format(
            #     precision_media) + " (average precision = " + '{0:0.2f}'.format(precision_avg_media) + ")")
            # if precision_media > ganador_metrica:
            #     ganador_metrica = precision_media
            #     ganador_metrica_avg = precision_avg_media
            #     ganador_nombreModelo = nombreModelo
            #     ganador_grid_mejores_parametros = modelo_grid_mejores_parametros
            #
            #
            # # ============================================================================
            # nombreModelo = "extra_trees_grid"
            # pathModelo = dir_subgrupo + nombreModelo + ".modelo"
            # # n_estimators=100, max_depth=11, min_samples_leaf=20, max_features=None, min_impurity_decrease=0.001, min_samples_split=3, random_state=1
            # modelo_base = ExtraTreesClassifier(n_estimators=ARBOLES_n_estimators, max_depth=ARBOLES_max_depth, min_samples_leaf=ARBOLES_min_samples_leaf,
            #                                           max_features=ARBOLES_max_features, min_samples_split=ARBOLES_min_samples_split,
            #                                           max_leaf_nodes=ARBOLES_max_leaf_nodes, min_impurity_decrease=ARBOLES_min_impurity_decrease)
            # # Tomados de https://predictivelearning.github.io/projects/Project_241_ML_AllEnsembleClassifiers_GaussianNB__Predict_Income_from_US_census.html
            # hiperparametros = {"n_estimators": [100], "criterion": ["gini"], "max_depth": [9, 11, 13], "min_impurity_decrease": [0.001, 0.00001],
            #                    "max_features": ["auto"], "min_samples_leaf": [15, 25], "min_samples_split": [3], "class_weight": [None]}
            # modelos_grid = GridSearchCV(modelo_base, hiperparametros, scoring='precision', n_jobs=-1, refit=True, return_train_score=False, cv=cv_todos)
            # modelo_grid_mejores_parametros = ejecutarModeloyGuardarlo(nombreModelo, modelos_grid, pathModelo, ds_train_f, ds_train_t, feature_names, True, modoDebug)
            # cargarModeloyUsarlo(dir_subgrupo_img, pathModelo, ds_test_f, ds_test_t, id_subgrupo, modoDebug)
            #
            # test_t_predicho = modelo.predict(ds_test_f);
            # validac_t_predicho = modelo.predict(ds_validac_f)
            # precision_test = precision_score(ds_test_t, test_t_predicho);
            # precision_avg_test = average_precision_score(ds_test_t, test_t_predicho)
            # precision_validation = precision_score(ds_validac_t, validac_t_predicho);
            # precision_avg_validation = average_precision_score(ds_validac_t, validac_t_predicho)
            # precision_media = (precision_test + precision_validation) / 2
            # precision_avg_media = (precision_avg_test + precision_avg_validation) / 2
            # print(id_subgrupo + " " + nombreModelo + " -> Precision = " + '{0:0.2f}'.format(
            #     precision_media) + " (average precision = " + '{0:0.2f}'.format(precision_avg_media) + ")")
            # if precision_media > ganador_metrica:
            #     ganador_metrica = precision_media
            #     ganador_metrica_avg = precision_avg_media
            #     ganador_nombreModelo = nombreModelo
            #     ganador_grid_mejores_parametros = modelo_grid_mejores_parametros

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

            # # Se calcula ENSEMBLE (mezcla de modelos)
            # estimators = [('logreg', modelo_logreg), ('rf_grid', modelo_rf_grid)]  # create our voting classifier, inputting our models
            # ensemble_model = VotingClassifier(estimators, voting='hard')
            #
            # # fit ensemble model to training data
            # ensemble_model.fit(ds_train_f, ds_train_t)  # test our model on the test data
            # pathModelo== dir_subgrupo + "ensemble" + ".modelo"
            # s = pickle.dump(ensemble_model, open(pathModelo, 'wb'))
            #
            # #Se pinta la precisión del ensemble
            # print("TEST ENSEMBLE -> Score (precision): " + '{0:0.2f}'.format(average_precision_score(ds_test_t, ensemble_model.predict(ds_test_f))))


elif (modoTiempo == "futuro" and pathCsvReducido.endswith('.csv') and os.path.isfile(pathCsvReducido) and os.stat(
        pathCsvReducido).st_size > 0):

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
        print("Numero de targets_predichos: " + str(len(targets_predichos)) + " con numero de TRUEs = " + str(
            np.sum(targets_predichos, where=["True"])))

        # probabilities
        probs = pd.DataFrame(data=modelo_predictor_ganador.predict_proba(inputFeatures_sinnulos),
                             index=inputFeatures_sinnulos.index)

        # UMBRAL MENOS PROBABLES CON TARGET=1. Cuando el target es 1, se guarda su probabilidad
        print(probs.columns)
        probabilidadesEnTargetUnoPeq = probs.iloc[:, 1]  # Cogemos solo la segunda columna: prob de que sea target=1
        probabilidadesEnTargetUnoPeq2 = probabilidadesEnTargetUnoPeq.apply(
            lambda x: x if (x >= umbralProbTargetTrue) else np.nan)
        probabilidadesEnTargetUnoPeq3 = probabilidadesEnTargetUnoPeq2[
            np.isnan(probabilidadesEnTargetUnoPeq2[:]) == False]  # Cogemos todos los no nulos
        probabilidadesEnTargetUnoPeq4 = probabilidadesEnTargetUnoPeq3.sort_values(ascending=False)
        numfilasSeleccionadas = int(granProbTargetUno * probabilidadesEnTargetUnoPeq4.shape[
            0] / 100)  # Como están ordenadas en descendente, cojo estas NUM primeras filas
        targets_predichosCorregidos_probs = probabilidadesEnTargetUnoPeq4[0:(numfilasSeleccionadas - 1)]
        targets_predichosCorregidos = targets_predichosCorregidos_probs.apply(lambda x: 1)

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
print("------------ FIN de capa 6----------------")
