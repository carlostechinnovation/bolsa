import datetime
import os.path
import pickle
import sys
import warnings
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from imblearn.combine import SMOTETomek
from pandas_profiling import ProfileReport
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.utils import resample
from tabulate import tabulate
from xgboost import XGBClassifier

from bolsa import C5C6ManualFunciones
from bolsa.C5C6ManualFunciones import pintarFuncionesDeDensidad


def entrenarModeloModoPasado(dir_subgrupo, ds_train_f, ds_train_t, ds_test_f, ds_test_t):
    """
    Entrena modelo predictivo (solo PASADO) y lo guarda en una ruta.
    :param dir_subgrupo:
    :param ds_train_f:
    :param ds_train_t:
    :param ds_test_f:
    :param ds_test_t:
    :return: Path absoluto del modelo entrenado guardado
    """
    print("PASADO - ENTRENANDO MODELO PREDICTIVO (entrenarModeloModoPasado)...")

    ################################# GENERACIÓN DE MODELOS #################################

    # ########################### MODELO SVC balanceado ##################################
    # print("Inicio de SVC")
    # modelo = SVC(class_weight='balanced',  # penalize
    #             probability=True)
    #
    # modelo=modelo.fit(ds_train_f, ds_train_t)
    # print("Fin de SVC")
    # nombreModelo = "svc"
    # ###########################################################################

    # ########################### MODELO MLP ##################################
    # # Resultado: 0.43
    # print("Inicio de MLP")
    # modelo = MLPClassifier()
    #
    # modelo=modelo.fit(ds_train_f, ds_train_t)
    # print("Fin de MLP")
    # nombreModelo = "mlp"
    # ###########################################################################

    # ########################## INICIO DE GRID SEARCH #########################################################
    #
    # print("Inicio de GRID SEARCH")
    #
    # skf = StratifiedKFold(n_splits=10)
    # clf = RandomForestClassifier(n_jobs=-1)
    # param_grid = {
    #     'min_samples_split': [3, 5, 10],
    #     'n_estimators': [100, 200],
    #     'max_depth': [3, 5, 15],
    #     'max_features': [3, 5, 10]
    # }
    # scorers = {
    #     'precision_score': make_scorer(precision_score),
    #     'recall_score': make_scorer(recall_score),
    #     'accuracy_score': make_scorer(accuracy_score)
    # }
    # grid = GridSearchCV(clf, param_grid=param_grid, scoring=scorers, refit='precision_score',
    #                     cv=skf, return_train_score=True, n_jobs=-1)
    #
    # # fit ensemble model to training data
    # modelo = grid.fit(ds_train_f, ds_train_t)  # test our model on the test data
    # nombreModelo = "gridsearch"
    #
    # print("Fin de GRID SEARCH")
    #
    # ########################## FIN DE GRID SEARCH #########################################################

    ########################## INICIO DE ENSEMBLE #########################################################

    # Se calcula ENSEMBLE (mezcla de modelos)
    # print("Inicio de Ensemble")
    # clf1 = SVC(class_weight='balanced',  # penalize
    #              probability=True)
    # #clf2 = RandomForestClassifier(class_weight='balanced') # Rendimiento: 0.45
    # clf2 = MLPClassifier()  # Rendimiento: 0.46
    # eclf = VotingClassifier(estimators=[('m1', clf1), ('m2', clf2)],
    #                 voting='soft')
    # params = {}
    # grid = GridSearchCV(estimator=eclf, param_grid=params, cv=cv_todos, scoring='precision', n_jobs=-1)
    #
    # # fit ensemble model to training data
    # modelo = grid.fit(ds_train_f, ds_train_t)  # test our model on the test data
    # nombreModelo = "ensemble"

    # print("Fin de Ensemble")

    ########################## FIN DE ENSEMBLE #########################################################

    # # ########################## INICIO DE XGBOOST OPTIMIZADO ########################################################33
    # #
    # #
    # #     #################### OPTIMIZACION DE PARAMETROS DE XGBOOST ###############################################################
    #
    # # Parametros por defecto de los modelos que usan árboles de decisión
    # ARBOLES_n_estimators = 80
    # ARBOLES_max_depth = 11
    # ARBOLES_min_samples_leaf = 20
    # ARBOLES_max_features = "auto"
    # ARBOLES_min_samples_split = 3
    # ARBOLES_max_leaf_nodes = None
    # ARBOLES_min_impurity_decrease = 0.001
    #
    # seed = 112  # Random seed
    #
    # # Descomentar para obtener los parámetros con optimización Bayesiana
    # # IMPORTANTE: se debe instalar el paquete de bayes en Conda: conda install -c conda-forge bayesian-optimization
    # # Se imprimirán en el log, pero debo luego meterlos manualmente en el modelo
    # # IMPORTANTE: DEBEN RELLENARSE 2 VALORES POR CADA ATRIBUTO DE PBOUND
    # # https://ayguno.github.io/curious/portfolio/bayesian_optimization.html
    #
    # print("Inicio del optimizador de parametros de XGBOOST...")
    #
    # # Parametros ordenados ALFABETICAMENTE porque la liberia lo obliga
    # pbounds = {
    #     'colsample_bytree': (0.1, 0.4),
    #     'gamma': (2, 10),
    #     'learning_rate': (0.2, 0.4),
    #     'max_delta_step': (0, 10),
    #     'max_depth': (4, 7),
    #     'min_child_weight': (2, 20),
    #     'n_estimators': (10, 50),
    #     'reg_alpha': (0.1, 0.8)
    # }
    #
    # hyperparameter_space = {
    # }
    #
    #
    # def xgboost_hyper_param(max_depth, learning_rate, n_estimators, reg_alpha, min_child_weight, colsample_bytree,
    #                         gamma, max_delta_step):
    #     """Crea un modelo XGBOOST con los parametros indicados en la entrada. Aplica el numero de iteraciones de cross-validation indicado
    #         """
    #     clf = XGBClassifier(colsample_bytree=colsample_bytree, gamma=gamma, learning_rate=learning_rate,
    #                         max_depth=int(max_depth), min_child_weight=int(min_child_weight),
    #                         n_estimators=int(n_estimators), reg_alpha=reg_alpha,
    #                         nthread=-1, objective='binary:logistic', seed=seed, use_label_encoder=False,
    #                         eval_metric=["map"], max_delta_step=max_delta_step, scale_pos_weight=1) #'logloss'
    #
    #     # Explicacion: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    #     return np.mean(cross_val_score(clf, ds_train_f, ds_train_t, cv=cv_todos, scoring='accuracy'))
    #
    #
    # # alpha is a parameter for the gaussian process
    # # Note that this is itself a hyperparameter that can be optimized.
    # gp_params = {"alpha": 1e-7}
    #
    # # LIBRERIA: https://github.com/fmfn/BayesianOptimization
    # # Parametros: https://github.com/fmfn/BayesianOptimization/blob/master/bayes_opt/bayesian_optimization.py
    # # Añadir carpeta dinámicamente: https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
    # sys.path.append('/bayes_opt')
    # from bayes_opt import BayesianOptimization
    #
    # optimizer = BayesianOptimization(f=xgboost_hyper_param, pbounds=pbounds, random_state=1,
    #                                  verbose=10)
    #
    # # Fichero de log JSON con los escenarios probados
    # # optimizacion_bayesiana_escenarios = "./optimiz_bayes_escenarios.json"
    # # bo_logger = JSONLogger(path=optimizacion_bayesiana_escenarios)
    # # optimizer.subscribe(Events.OPTIMIZATION_STEP, bo_logger)
    # # if os.path.isfile(optimizacion_bayesiana_escenarios):  # Si ya existe una lista de puntos previa, los precargo
    # #     load_logs(optimizer, logs=[optimizacion_bayesiana_escenarios]);
    # #     print("New optimizer is now aware of {} points.".format(len(optimizer.space)))
    #
    # # print("Optimización de procesos bayesianos - Añadimos ESCENARIOS CONCRETOS para fijarlos (tuplas de parametros) que hayamos visto que tienen buenos resultados...")
    # # optimizer.probe(params={"colsample_bytree": 0.4, "gamma": 2.0, "learning_rate": 0.4, "max_delta_step": 9.6, "max_depth": 7.0, "min_child_weight": 8.2, "n_estimators": 47, "reg_alpha": 0.1}, lazy=False)
    #
    # optimizer.maximize(init_points=5, n_iter=20, acq='ucb', kappa=30, **gp_params)
    # #optimizer.maximize(init_points=5, n_iter=20, acq='poi', kappa=3, **gp_params)
    # # KAPPA: Parameter to indicate how closed are the next parameters sampled
    #
    # valoresOptimizados = optimizer.max
    # print(valoresOptimizados)
    # print("Fin del optimizador")
    # ###################################################################################
    #
    # print("Inicio de XGBOOST")
    # nombreModelo = "xgboost"
    # pathModelo = dir_subgrupo + nombreModelo + ".modelo"
    # # Parametros: https://xgboost.readthedocs.io/en/latest/parameter.html
    # # Instalación en Conda: conda install -c anaconda py-xgboost
    # # Instalación en Python básico: pip install xgboost
    #
    # # MODELO LUIS AUTOOPTIMIZADO PARA CADA SUBGRUPO
    # max_depth = int(valoresOptimizados.get("params").get("max_depth"))
    # learning_rate = valoresOptimizados.get("params").get("learning_rate")
    # n_estimators = int(valoresOptimizados.get("params").get("n_estimators"))
    # reg_alpha = valoresOptimizados.get("params").get("reg_alpha")
    # min_child_weight = int(valoresOptimizados.get("params").get("min_child_weight"))
    # colsample_bytree = valoresOptimizados.get("params").get("colsample_bytree")
    # gamma = valoresOptimizados.get("params").get("gamma")
    # max_delta_step = valoresOptimizados.get("params").get("max_delta_step")
    # nthread = -1
    # objective = 'binary:logistic'
    # seed = seed
    #
    # modelo = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators,
    #                        reg_alpha=reg_alpha, min_child_weight=min_child_weight,
    #                        colsample_bytree=colsample_bytree, gamma=gamma,
    #                        nthread=nthread, objective=objective, seed=seed, use_label_encoder=False,
    #                        max_delta_step=max_delta_step, scale_pos_weight=1)
    #
    # eval_set = [(ds_train_f.to_numpy(), ds_train_t.to_numpy().ravel()), (ds_test_f, ds_test_t)]
    # modelo = modelo.fit(ds_train_f.to_numpy(), ds_train_t.to_numpy().ravel(), eval_metric=["map"],
    #                     early_stopping_rounds=3, eval_set=eval_set,
    #                     verbose=False)  # ENTRENAMIENTO (TRAIN)
    #
    # # ########################## FIN DE XGBOOST OPTIMIZADO ########################################################

    ###################### MODELO LGBM ######################
    from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold, cross_validate, StratifiedShuffleSplit
    from sklearn import metrics
    import lightgbm as lgb

    nombreModelo = "lgbm"

    params = {'objective': 'binary',
              'learning_rate': 0.01,
              "boosting_type": "gbdt",
              "metric": 'precision',
              'n_jobs': -1,
              'min_data_in_leaf': 5,
              'min_child_samples': 5,
              'num_leaves': 20,  # maximo numero de hojas
              'max_depth': 1,
              'random_state': 0,
              'importance_type': 'split',
              'min_split_gain': 0.0,
              # min_child_weight = 0.001,  subsample = 1.0, subsample_freq = 0, colsample_bytree = 1.0, reg_alpha = 0.0, reg_lambda = 0.0,
              }

    modelo = lgb.LGBMClassifier(**params, n_estimators=200)
    modelo.fit(ds_train_f, ds_train_t, eval_set=[(ds_train_f, ds_train_t), (ds_test_f, ds_test_t)])
    #####################################################

    # ########################## INICIO DE XGBOOST SIN OPTIMIZAR ########################################################
    #
    # nombreModelo = "xgboost_noopt"
    #
    # modelo = XGBClassifier()
    #
    # eval_set = [(ds_train_f, ds_train_t), (ds_test_f, ds_test_t)]
    # modelo = modelo.fit(ds_train_f, ds_train_t, eval_metric=["map"], early_stopping_rounds=4, eval_set=eval_set,
    #                     verbose=False)  # ENTRENAMIENTO (TRAIN)
    #
    # ########################## FIN DE XGBOOST SIN OPTIMIZAR ########################################################

    ############################## GUARDADO DE MODELO #########################
    pathModelo = dir_subgrupo + nombreModelo + ".modelo"
    pickle.dump(modelo, open(pathModelo, 'wb'))

    print("PASADO - entrenarModeloModoPasado - GUARDANDO MODELO PREDICTIVO ENTRENADO EN: " + pathModelo)

    return pathModelo, nombreModelo


def fitNormalizadorDeFeaturesSoloModoPasado(featuresDF, path_modelo_normalizador):
    """
    MODO PASADO - Entrena el modelo normalizador de features y lo guarda en una ruta.
    Ese modelo puede usarse después para normalizar esas mismas features del pasado u otras del futuro.
    :param featuresDF:
    :param path_modelo_normalizador:
    :return:
    """
    # Con el "normalizador COMPLEJO" solucionamos este bug: https://github.com/scikit-learn/scikit-learn/issues/14959  --> Aplicar los cambios indicados a:_/home/carloslinux/Desktop/PROGRAMAS/anaconda3/envs/BolsaPython/lib/python3.7/site-packages/sklearn/preprocessing/_data.py
    modelo_normalizador = make_pipeline(MinMaxScaler(), PowerTransformer(method='yeo-johnson', standardize=True, copy=True), ).fit(featuresDF)  # COMPLEJO
    # modelo_normalizador = PowerTransformer(method='yeo-johnson', standardize=True, copy=True).fit(featuresFichero)
    pickle.dump(modelo_normalizador, open(path_modelo_normalizador, 'wb'))


def normalizar(path_modelo_normalizador, featuresFichero, modoTiempo, pathCsvIntermedio, modoDebug, dir_subgrupo_img, dibujoBins, DEBUG_FILTRO):
    """
    MODOS PASADO Y FUTURO - Aplica la NORMALIZACIÓN
    :param path_modelo_normalizador:
    :param featuresFichero:
    :param modoTiempo:
    :param pathCsvIntermedio:
    :param modoDebug:
    :param dir_subgrupo_img:
    :param dibujoBins:
    :param DEBUG_FILTRO:
    :return: Dataframe con las columnas normalizadas.
    """
    print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " ----- NORMALIZACIÓN de las features ------")
    print("NORMALIZACION: hacemos que todas las features tengan distribución gaussiana media 0 y varianza 1. El target no se toca.")
    print("featuresFichero: " + str(featuresFichero.shape[0]) + " x " + str(featuresFichero.shape[1]))
    print("pathCsvIntermedio: " + pathCsvIntermedio)
    print("path_modelo_normalizador: " + path_modelo_normalizador)

    # Vamos a normalizar z-score (media 0, std_dvt=1), pero yeo-johnson tiene un bug (https://github.com/scipy/scipy/issues/10821) que se soluciona sumando una constante a toda la matriz, lo cual no afecta a la matriz normalizada
    featuresFichero = featuresFichero + 1.015815

    if modoTiempo == "pasado":
        fitNormalizadorDeFeaturesSoloModoPasado(featuresFichero, path_modelo_normalizador)

    # Pasado o futuro: Cargar normalizador
    modelo_normalizador = pickle.load(open(path_modelo_normalizador, 'rb'))
    print("Aplicando normalizacion, manteniendo indices y nombres de columnas...")
    featuresFicheroNorm = pd.DataFrame(data=modelo_normalizador.transform(featuresFichero), index=featuresFichero.index, columns=featuresFichero.columns)
    print("featuresFicheroNorm:" + str(featuresFicheroNorm.shape[0]) + " x " + str(featuresFicheroNorm.shape[1]))
    C5C6ManualFunciones.mostrarEmpresaConcretaConFilter(featuresFicheroNorm, DEBUG_FILTRO, "featuresFicheroNorm")
    featuresFicheroNorm.to_csv(pathCsvIntermedio + ".normalizado.csv", index=True, sep='|', float_format='%.4f')  # UTIL para testIntegracion

    if modoDebug and modoTiempo == "pasado":
        pintarFuncionesDeDensidad(featuresFichero, dir_subgrupo_img, dibujoBins, "normalizadas")

    return featuresFicheroNorm


def comprobarSuficientesClasesDelTarget(targetsFichero, modoTiempo):
    """
    Comprueba el numero de clases diferentes y aborta en caso de que no haya suficientes.
    :param targetsFichero:
    :param modoTiempo:
    :return: Numero de clases encontradas
    """
    print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " ----- comprobarSuficientesClasesDelTarget ------")
    y_unicos = np.unique(targetsFichero)
    print("Clases encontradas en el target: ")
    print(y_unicos)
    numclases = y_unicos.size

    if modoTiempo == "pasado" and numclases <= 1:
        print(modoTiempo + " - El subgrupo solo tiene " + str(numclases) + " clases en el target. No son SUFICIENTES. Abortamos...")
        exit(-1)

    return numclases


def matrizCorrelacionesYquitarFaturesCorreladas(ift_juntas, umbralFeaturesCorrelacionadas, pathListaColumnasCorreladasDrop, pathFeaturesSeleccionadas):
    """
    Matriz de correlaciones y quitar features correladas
    :param ift_juntas:
    :param umbralFeaturesCorrelacionadas:
    :param pathListaColumnasCorreladasDrop:
    :param pathFeaturesSeleccionadas:
    :return:
    """
    print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " Matriz de correlaciones (PASADO):")
    matrizCorr = ift_juntas.corr().abs()
    # print(matrizCorr.to_string())
    upper = matrizCorr.where(np.triu(np.ones(matrizCorr.shape), k=1).astype(bool))
    # print(upper.to_string())
    print("Eliminamos las features muy correladas (umbral = " + str(umbralFeaturesCorrelacionadas) + "):")
    to_drop = [column for column in upper.columns if any(upper[column] > umbralFeaturesCorrelacionadas)]
    print(to_drop)
    print("Guardamos esa lista de features muy correladas en: " + pathListaColumnasCorreladasDrop)
    pickle.dump(to_drop, open(pathListaColumnasCorreladasDrop, 'wb'))
    ift_juntas.drop(to_drop, axis=1, inplace=True)
    # print(ift_juntas)
    print("Matriz de correlaciones corregida, habiendo quitado las muy correlacionadas (PASADO):")
    matrizCorr = ift_juntas.corr()
    # print(matrizCorr)
    print("matrizCorr:" + str(matrizCorr.shape[0]) + " x " + str(matrizCorr.shape[1]))
    # ##################################################################
    columnasSeleccionadas = ift_juntas.columns
    print("Guardando las columnas seleccionadas en: ", pathFeaturesSeleccionadas)
    # print(columnasSeleccionadas)
    columnasSeleccionadasStr = '|'.join(columnasSeleccionadas)
    featuresSeleccionadasFile = open(pathFeaturesSeleccionadas, "w")
    featuresSeleccionadasFile.write(columnasSeleccionadasStr)
    featuresSeleccionadasFile.close()


def pintarDistribucionProbabDelTargetPredicho(dir_subgrupo_img, nombreFichero, modeloPredictivoEntrenado, featuresDF, id_subgrupo):
    """
    Para el target predicho, pinta la distribución de probabilidades (predict_proba) y la guarda en una imagen
    :param dir_subgrupo_img:
    :param nombreFichero:
    :param modeloPredictivoEntrenado:
    :param featuresDF:
    :param id_subgrupo:
    :return:
    """
    print("pintarDistribucionProbabDelTargetPredicho-INICIO")
    path_dibujo_probabs = dir_subgrupo_img + nombreFichero
    print("Distribución de las probabilidades del target predicho (debe ser con forma de U para que distinga bien los positivos de los negativos): " + path_dibujo_probabs)
    probsPositivosyNegativos = modeloPredictivoEntrenado.predict_proba(featuresDF)

    # probabilities
    probs = pd.DataFrame(data=probsPositivosyNegativos)
    probs.columns = ['proba_false', 'proba_true']
    print("PASADO - Ejemplos de probabilidades al predecir targets true (orden descendiente): ")
    print(tabulate(probs.sort_values("proba_true", ascending=False).head(n=5), headers='keys', tablefmt='psql'))
    print("PASADO - Ejemplos de probabilidades al predecir targets true (orden ascendiente): ")
    print(tabulate(probs.sort_values("proba_true", ascending=True).head(n=5), headers='keys', tablefmt='psql'))

    probsPositivosyNegativosDF = pd.DataFrame(data=probsPositivosyNegativos, columns=["probabsNegativas", "probabsPositivas"])
    plt.hist(probsPositivosyNegativosDF.iloc[:, 1], label="Probabilidades target predicho", bins=20, alpha=.7,
             color='blue')
    plt.title("Distribución de la probab predicha al predecir el target (subgrupo " + id_subgrupo + ")",
              fontsize=10)
    plt.legend(loc='upper left')
    plt.savefig(path_dibujo_probabs, bbox_inches='tight')
    plt.clf();
    plt.cla();
    plt.close()  # Limpiando dibujo
    print("pintarDistribucionProbabDelTargetPredicho-FIN")


def calcularMetricasModeloEntrenado(id_subgrupo, modeloPredictivoEntrenado, ds_train_f_sinsmote, ds_train_t_sinsmote, ds_train_t, ds_test_f, ds_test_t, ds_validac_f, ds_validac_t, dir_subgrupo_img,
                                    pathCsvIntermedio, tasaDesbalanceoAntes, nombreModelo, ganador_metrica):
    """

    :param id_subgrupo:
    :param modeloPredictivoEntrenado:
    :param ds_train_f_sinsmote:
    :param ds_train_t_sinsmote:
    :param ds_train_t:
    :param ds_test_f:
    :param ds_test_t:
    :param ds_validac_f:
    :param ds_validac_t:
    :param dir_subgrupo_img:
    :param pathCsvIntermedio:
    :param tasaDesbalanceoAntes:
    :param nombreModelo:
    :param ganador_metrica:
    :return:
    """
    print("Inicio de ANÁLISIS DE RESULTADOS (calcularMetricasModeloEntrenado) - train vs test vs validación")

    print("ds_train_f_sinsmote: " + str(ds_train_f_sinsmote.shape[0]) + " x " + str(ds_train_f_sinsmote.shape[1]))
    train_t_predicho = modeloPredictivoEntrenado.predict(ds_train_f_sinsmote)
    precision_train = precision_score(ds_train_t_sinsmote, train_t_predicho)
    pintarDistribucionProbabDelTargetPredicho(dir_subgrupo_img, "histograma_target_probabilidades.png", modeloPredictivoEntrenado, ds_train_f_sinsmote, id_subgrupo)

    print("Informe de metricas:")
    print(classification_report(ds_train_t_sinsmote, train_t_predicho))

    if precision_train == 0:
        # print(train_t_predicho)
        raise NameError("La precision calculada es 0 porque posiblemente no tenemos casos positivos en la muestra predicha. Salimos del proceso de este subgrupo " + id_subgrupo + "...")

    print("PRECISIÓN EN TRAIN = " + '{0:0.2f}'.format(precision_train))
    pd.DataFrame(ds_train_t).to_csv(pathCsvIntermedio + ".ds_train_t_sinsmote.csv", index=True, sep='|', float_format='%.4f')  # NO BORRAR: UTIL para testIntegracion
    pd.DataFrame(train_t_predicho).to_csv(pathCsvIntermedio + ".train_t_predicho.csv", index=True, sep='|', float_format='%.4f')  # NO BORRAR: UTIL para testIntegracion

    print("ds_validac_f: " + str(ds_validac_f.shape[0]) + " x " + str(ds_validac_f.shape[1]))
    print("ds_validac_t: " + str(len(ds_validac_t)))

    ds_test_t_pred = modeloPredictivoEntrenado.predict(ds_test_f)
    test_t_predicho = ds_test_t_pred
    validac_t_predicho = modeloPredictivoEntrenado.predict(ds_validac_f)
    precision_test = precision_score(ds_test_t, test_t_predicho)
    precision_avg_test = average_precision_score(ds_test_t, test_t_predicho)
    precision_validation = precision_score(ds_validac_t, validac_t_predicho)
    precision_avg_validation = average_precision_score(ds_validac_t, validac_t_predicho)
    precision_media = (precision_test + precision_validation) / 2
    precision_avg_media = (precision_avg_test + precision_avg_validation) / 2
    print("PRECISIÓN EN TEST = " + '{0:0.2f}'.format(precision_test))
    print("PRECISIÓN EN VALIDACION = " + '{0:0.2f}'.format(precision_validation))

    # NO BORRAR: UTIL para informe HTML entregable
    precisionSistemaRandom = 1 / tasaDesbalanceoAntes  # Precision del sistema tonto
    precisionMediaTestValid = (precision_test + precision_validation) / 2
    mejoraRespectoSistemaRandom = 100 * (precisionMediaTestValid - precisionSistemaRandom) / precisionSistemaRandom
    print("ENTREGABLEPRECISIONESPASADO"
          + "|id_subgrupo:" + str(id_subgrupo)
          + "|precisionpasadotrain:" + str(round(precision_train, 2))
          + "|precisionpasadotest:" + str(round(precision_test, 2))
          + "|precisionpasadovalidacion:" + str(round(precision_validation, 2))
          + "|precisionsistemarandom:" + str(round(precisionSistemaRandom, 2))
          + "|mejoraRespectoSistemaRandom:" + str(round(mejoraRespectoSistemaRandom, 0)) + " %"
          )

    # NO BORRAR: UTIL para testIntegracion
    pd.DataFrame(ds_test_t).to_csv(pathCsvIntermedio + ".ds_test_t.csv", index=True, sep='|', float_format='%.4f')
    pd.DataFrame(test_t_predicho).to_csv(pathCsvIntermedio + ".test_t_predicho.csv", index=True, sep='|', float_format='%.4f')
    pd.DataFrame(ds_validac_t).to_csv(pathCsvIntermedio + ".ds_validac_t.csv", index=True, sep='|', float_format='%.4f')
    pd.DataFrame(validac_t_predicho).to_csv(pathCsvIntermedio + ".validac_t_predicho.csv", index=True, sep='|', float_format='%.4f')

    print(id_subgrupo + " " + nombreModelo + " -> Precision = " + '{0:0.2f}'.format(precision_media) + " (average precision = " + '{0:0.2f}'.format(precision_avg_media) + ")")
    if precision_media > ganador_metrica:
        ganador_metrica = precision_media
        ganador_metrica_avg = precision_avg_media
        ganador_nombreModelo = nombreModelo
        ganador_grid_mejores_parametros = []  # Este modelo no tiene esta variable

    print("Fin de ANÁLISIS DE RESULTADOS (calcularMetricasModeloEntrenado)")

    return train_t_predicho, test_t_predicho, validac_t_predicho, ganador_metrica, ganador_metrica_avg, ganador_nombreModelo, ganador_grid_mejores_parametros
