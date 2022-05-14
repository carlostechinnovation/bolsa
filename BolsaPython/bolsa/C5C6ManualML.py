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

'''
Entrena modelo predictivo (solo PASADO) y lo guarda en una ruta.
@:returns Path absoluto del modelo entrenado guardado 
'''
def entrenarModeloModoPasado(dir_subgrupo, ds_train_f, ds_train_t, ds_test_f, ds_test_t):

    print("PASADO - ENTRENANDO MODELO PREDICTIVO...")

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
              'learning_rate': 0.08,
              "boosting_type": "gbdt",
              "metric": 'precision',
              'n_jobs': -1,
              'min_data_in_leaf': 6,
              'min_child_samples': 3,
              'num_leaves': 15,  # maximo numero de hojas
              'max_depth': 4,
              'random_state': 0,
              'importance_type': 'split',
              'min_split_gain': 0.0,
              # min_child_weight = 0.001,  subsample = 1.0, subsample_freq = 0, colsample_bytree = 1.0, reg_alpha = 0.0, reg_lambda = 0.0,
              }

    modelo = lgb.LGBMClassifier(**params, n_estimators=50)
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

    print("PASADO - GUARDANDO MODELO PREDICTIVO ENTENADO EN: " + pathModelo)

    return pathModelo, nombreModelo