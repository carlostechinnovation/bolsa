import sys

import h2o as h2o
import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours, NearMiss, NeighbourhoodCleaningRule
from numpy import mean
from scipy.stats import stats, chi2
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.fixes import loguniform
from pathlib import Path
from sklearn import svm
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, recall_score, make_scorer, \
    precision_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score, train_test_split, \
    ParameterGrid
from sklearn.calibration import CalibratedClassifierCV
from shutil import copyfile
import os.path
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.tree import export_graphviz
from subprocess import call
from xgboost import XGBClassifier
from matplotlib import pyplot
import datetime
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import time
import matplotlib.patches as mpatches
from h2o.automl import H2OAutoML

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
fraccion_train = 0.75  # Fracción de datos usada para entrenar
fraccion_test = 0.15  # Fracción de datos usada para testear (no es validación)
fraccion_valid = 1-(fraccion_train + fraccion_test)

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
umbralProbTargetTrue = float("0.2")

print("dir_subgrupo: %s" % dir_subgrupo)
print("modoTiempo: %s" % modoTiempo)
print("desplazamientoAntiguedad: %s" % desplazamientoAntiguedad)
print("pathCsvReducido: %s" % pathCsvReducido)
print("dir_subgrupo_img = %s" % dir_subgrupo_img)
print("umbralProbTargetTrue = " + str(umbralProbTargetTrue))
print("balancearConSmoteSoloTrain = " + str(balancearConSmoteSoloTrain))
print("umbralFeaturesCorrelacionadas = " + str(umbralFeaturesCorrelacionadas))


################# MAIN ########################################
# GANADOR DEL SUBGRUPO (acumuladores)
ganador_nombreModelo = "NINGUNO"
ganador_metrica = 0
ganador_metrica_avg = 0
pathListaColumnasCorreladasDrop = (dir_subgrupo + "columnas_correladas_drop" + ".txt").replace("futuro",
                                                                                               "pasado")  # lo guardamos siempre en el pasado

if (modoTiempo == "pasado" and pathCsvCompleto.endswith('.csv') and os.path.isfile(pathCsvCompleto) and os.stat(
        pathCsvCompleto).st_size > 0):

    print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " Capa 6 - Modo PASADO")

    inputFeaturesyTarget = pd.read_csv(pathCsvCompleto, index_col=0, sep='|')  # La columna 0 contiene el indice

    print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " DIVIDIR EL DATASET DE ENTRADA EN 3 PARTES: TRAIN ("+str(fraccion_train)+"), TEST ("+str(fraccion_test)+"), VALIDACION ("+str(fraccion_valid)+")")
    ds_train, ds_test, ds_validacion = np.split(inputFeaturesyTarget.sample(frac=1),
                                                [int(fraccion_train * len(inputFeaturesyTarget)),
                                                 int((fraccion_train+fraccion_test) * len(inputFeaturesyTarget))])
    print("TRAIN = " + str(ds_train.shape[0]) + " x " + str(ds_train.shape[1]) + "  " + "TEST --> " + str(
        ds_test.shape[0]) + " x " + str(ds_test.shape[1]) + "  " + "VALIDACION --> " + str(
        ds_validacion.shape[0]) + " x " + str(ds_validacion.shape[1]))

    print("Separamos FEATURES y TARGETS, de los 3 dataframes...")
    ds_train_f = ds_train.drop('TARGET', axis=1).to_numpy()
    ds_train_t = ds_train[['TARGET']].to_numpy().ravel()
    ds_test_f = ds_test.drop('TARGET', axis=1).to_numpy()
    ds_test_t = ds_test[['TARGET']].to_numpy().ravel()
    ds_validac_f = ds_validacion.drop('TARGET', axis=1).to_numpy()
    ds_validac_t = ds_validacion[['TARGET']].to_numpy().ravel()

    feature_names = ds_train.columns.drop('TARGET')


#----------------------INICIO DE H20-------------------------------------------
    # Imports para AutoML:
    import h2o
    import seaborn as sns
    from h2o.automl import H2OAutoML
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.gridspec as gridspec
    from sklearn.metrics import confusion_matrix, plot_confusion_matrix

    # Start cluster:
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    # Arranque de H2O
    h2o.init()

    # Se eliminan las filas con missing values (se necesita así en XGBoost)
    inputFeaturesyTarget = inputFeaturesyTarget.dropna()

    # División del dataset
    ds_train, ds_test, ds_validacion = np.split(inputFeaturesyTarget.sample(frac=1),
                                       [int(fraccion_train * len(inputFeaturesyTarget)),
                                        int((fraccion_train + fraccion_test) * len(inputFeaturesyTarget))])

    # Conversión para h2o frame:
    traindf = h2o.H2OFrame(ds_train)
    testdf = h2o.H2OFrame(ds_test)
    validaciondf = h2o.H2OFrame(ds_validacion)

    # Cración de las variables de entrada en el AutoML:
    y = "TARGET"
    x = list(traindf.columns)
    x.remove(y)

    # Tratamiento de la variable target
    traindf[y] = traindf[y].asfactor()
    testdf[y] = testdf[y].asfactor()
    validaciondf[y] = validaciondf[y].asfactor()

    # AutoML H2O:
    aml = H2OAutoML(max_models=80, max_runtime_secs=1*360, seed=247)  # --> CAMBIAR EL TIEMPO a 1*3600
    aml.train(x=x, y=y, training_frame=traindf)

    # Leader board:
    print(aml.leaderboard)

    # Se guardan los modelos que comiencen por "StackedEnsemble_AllModels_AutoML_"
    lb = aml.leaderboard
    model_ids = list(lb['model_id'].as_data_frame().iloc[:, 0])
    for m_id in model_ids:
        if str(m_id).startswith("StackedEnsemble_AllModels_AutoML_"):
            mdl = h2o.get_model(m_id)
            h2o.save_model(model=mdl, path=dir_subgrupo, force=True)
    # Se guarda también el leaderboard
    h2o.export_file(lb, os.path.join(dir_subgrupo, 'aml_leaderboard.h2o'), force=True)

    # Se carga el primer modelo de entre los que comienzan por 'StackedEnsemble_AllModels_AutoML_'
    models_path = dir_subgrupo
    lb = h2o.import_file(path=os.path.join(models_path, "aml_leaderboard.h2o"))
    modelosBest = [filename for filename in os.listdir(models_path) if
                   filename.startswith("StackedEnsemble_AllModels_AutoML_")]
    modeloSeleccionado = modelosBest[0]
    print("MODELO USADO: " + modeloSeleccionado)
    modelo = h2o.load_model(os.path.join(models_path, modeloSeleccionado))

    #Se guarda el nombre del modelo para el futuro
    ganador_nombreModelo=str(modelo.model_id)

    # Predicciones y tratamientos
    test_predicho_h20=modelo.predict(testdf).as_data_frame()
    validac_predicho_h20 = modelo.predict(validaciondf).as_data_frame()

    # Conversión a dataframe pandas
    tmp1=ds_test['TARGET'].reset_index(drop=True)
    tmp2=test_predicho_h20['predict'].reset_index(drop=True)
    # print('tmp1')
    # print(tmp1.head())
    # print('tmp2')
    # print(tmp2.head())
    df_test = pd.concat([tmp1, tmp2], axis=1)
    df_test.columns = ['actual', 'Ypredict']

    tmp1 = ds_validacion['TARGET'].reset_index(drop=True)
    tmp2 = validac_predicho_h20['predict'].reset_index(drop=True)
    df_validacion = pd.concat([tmp1, tmp2], axis=1)
    df_validacion.columns = ['actual', 'Ypredict']

    # df_test = pd.DataFrame({'actual': ds_test['TARGET'], 'Ypredict': test_predicho_h20['predict']})
    # df_validacion = pd.DataFrame({'actual': ds_validacion['TARGET'], 'Ypredict': validac_predicho_h20['predict']})

    # Matriz de confusión
    from sklearn.metrics import confusion_matrix
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(df_test['actual'], df_test['Ypredict']).ravel()
    tn_val, fp_val, fn_val, tp_val = confusion_matrix(df_validacion['actual'], df_validacion['Ypredict']).ravel()
    print("MATRIZ DE CONFUSIÓN test: ")
    print("tn_test, fp_test, fn_test, tp_test: ")
    print(tn_test, fp_test, fn_test, tp_test)
    precision_test=tp_test / (tp_test + fp_test)
    recall_test=tp_test / (tp_test + fn_test)
    print("Precision: " + str(precision_test))
    print("Recall: " + str(recall_test))
    print("MATRIZ DE CONFUSIÓN validación: ")
    print("tn_val, fp_val, fn_val, tp_val: ")
    print(tn_val, fp_val, fn_val, tp_val)
    precision_val = tp_val / (tp_val + fp_val)
    recall_val = tp_val / (tp_val + fn_val)
    print("Precision: " + str(precision_val))
    print("Recall: " + str(recall_val))
    precision_media = (precision_test + precision_val) / 2
    print("PRECISION MEDIA: "+str(precision_media))

    precision_test = precision_score(df_test['actual'], df_test['Ypredict'])
    precision_avg_test = average_precision_score(df_test['actual'], df_test['Ypredict'])
    precision_validation = precision_score(df_validacion['actual'], df_validacion['Ypredict'])
    precision_avg_validation = average_precision_score(df_validacion['actual'], df_validacion['Ypredict']);

    precision_media = (precision_test + precision_validation) / 2
    precision_avg_media = (precision_avg_test + precision_avg_validation) / 2
    print(id_subgrupo + " " + ganador_nombreModelo + " -> Precision = " + '{0:0.2f}'.format(
        precision_media) + " (average precision = " + '{0:0.2f}'.format(precision_avg_media) + ")")
    ganador_metrica = precision_media
    ganador_metrica_avg = precision_avg_media
    ganador_nombreModelo = ganador_nombreModelo

#-------------------------------FIN DE H2O----------------------------------

    print("********* GANADOR de subgrupo *************")
    print("PASADO -> " + id_subgrupo + " (num features = " + str(
        ds_train_f.shape[1]) + ")" + " -> Modelo ganador = " + ganador_nombreModelo + " --> METRICA = " + str(
        round(ganador_metrica, 4)) + " (avg_precision = " + str(round(ganador_metrica_avg, 4)) + ") ")
    pathModeloGanadorDeSubgrupoOrigen = dir_subgrupo + ganador_nombreModelo
    pathModeloGanadorDeSubgrupoDestino = pathModeloGanadorDeSubgrupoOrigen + "_ganador"
    print(pathModeloGanadorDeSubgrupoOrigen)
    print(pathModeloGanadorDeSubgrupoDestino)
    copyfile(pathModeloGanadorDeSubgrupoOrigen, pathModeloGanadorDeSubgrupoDestino)
    print("Modelo ganador guardado en: " + pathModeloGanadorDeSubgrupoDestino)

    ######################################################################################################################
    ######################################################################################################################


elif (modoTiempo == "futuro" and pathCsvCompleto.endswith('.csv') and os.path.isfile(pathCsvCompleto) and os.stat(
        pathCsvCompleto).st_size > 0):

    print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") + " Capa 6 - Modo futuro")
    inputFeaturesyTarget = pd.read_csv(pathCsvCompleto, index_col=0, sep='|')
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
            # Se coge el último modelo que se apellida "ganador" (se itera todo el directorio)
            path_modelo_predictor_ganador = os.path.join(dir_modelo_predictor_ganador, file)
            print("MODELO USADO: " + path_modelo_predictor_ganador)

    print("Cargar modelo PREDICTOR ganador (de la carpeta del pasado, SI EXISTE): " + path_modelo_predictor_ganador)
    if os.path.isfile(path_modelo_predictor_ganador):

        # Arranque de H2O
        h2o.init()
        # Se carga el modelo
        modelo = h2o.load_model(path_modelo_predictor_ganador)

        # Limpieza de datos
        print(
            "MISSING VALUES (FILAS) - Borramos las FILAS que tengan 1 o mas valores NaN porque son huecos que no deberían estar...")
        inputFeatures_sinnulos = inputFeatures.dropna(axis=0, how='any')  # Borrar FILA si ALGUNO sus valores tienen NaN

        # Predicciones y tratamientos
        print("Predecir:")
        predict = modelo.predict(h2o.H2OFrame(inputFeatures_sinnulos))
        p = predict.as_data_frame()

        # Conversión a dataframe pandas
        targets_predichos=pd.DataFrame(p['predict'].tolist(), columns=['TARGET_PREDICHO'])

        # print("Numero de targets_predichos: " + str(len(targets_predichos)) + " con numero de TRUEs = " + str(
        #     np.sum(targets_predichos, where=["1"])))

        # probabilities
        probs = pd.concat([p['p0'], p['p1']], axis=1)
        probs.columns = ['0', '1']
        print("probs: ")
        print(probs.head())


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


# Shutdown h2o cluster:
h2o.cluster().shutdown(prompt=False)


############################################################
print((datetime.datetime.now()).strftime("%Y%m%d_%H%M%S") +" ------------ FIN de capa H2O----------------")
