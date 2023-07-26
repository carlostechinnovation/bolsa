##### Regression
import shutil

import autosklearn.classification
import numpy as np
import pandas as pd
import sklearn.metrics

if __name__ == "__main__":  # Esto sirve para que el paralelizador DASK sepa cual es el codigo principal de cada hilo/thread
    print("-------- PARAMETROS -------- ")
    PATH_CSV_ENTRADA = "/bolsa/pasado/subgrupos/SG_11/COMPLETO.csv"
    DATASETNAME = "SG11pasado"

    print("-------- Limpieza -------- ")
    shutil.rmtree(path="/tmp/autosklearn_classification_example_tmp", ignore_errors=True)
    shutil.rmtree(path="/tmp/autosklearn_classification_example_out", ignore_errors=True)

    print("-------- Leer dataset de entrada -------- ")
    entradaDF = pd.read_csv(PATH_CSV_ENTRADA, index_col=None, sep='|')  # La columna 0 contiene el indice

    print("MISSING VALUES (FILAS) - Borramos las FILAS que tengan 1 o mas valores NaN porque son huecos que no deberían estar...")
    entradaDF2 = entradaDF.dropna(axis=0, how='any')  # Borrar FILA si ALGUNO sus valores tienen NaN

    print("Borrar columnas especiales (idenficadoras de fila): empresa | antiguedad | mercado | anio | mes | dia | hora | minuto...")
    entradaDF3 = entradaDF2.drop('empresa', axis=1).drop('antiguedad', axis=1).drop('mercado', axis=1) .drop('anio', axis=1).drop('mes', axis=1).drop('dia', axis=1).drop('hora', axis=1).drop('minuto', axis=1)

    X = entradaDF3.drop('TARGET', axis=1).to_numpy()
    y = entradaDF3[['TARGET']].to_numpy().ravel()

    print("-------- Partir el dataset en TRAIN y TEST -------- ")
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)
    X_train = X_train.astype(np.float)
    y_train = y_train.astype(np.int)

    print("-------- Crear modelo y entrenarlo -------- ")
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=600,
        ensemble_size=50,
        ensemble_nbest=50,
        max_models_on_disc=100,
        seed=1,
        memory_limit=(7 * 1024),
        tmp_folder=None,
        output_folder=None,
        delete_tmp_folder_after_terminate=True,
        delete_output_folder_after_terminate=True,
        n_jobs=5,
        dask_client=None,
        disable_evaluator_output=False,
        smac_scenario_args=None,
        logging_config=None,
        metric=autosklearn.metrics.average_precision,
        scoring_functions=None,
        load_models=True,
    )

    automl.fit(X_train, y_train, dataset_name=DATASETNAME)

    print("-------- Print the final ensemble constructed by auto-sklearn -------- ")
    print(automl.show_models())

    # print("-------- METRICAS DISPONIBLES -------- ")
    # print("Available CLASSIFICATION metrics autosklearn.metrics.*:")
    # print("\t*" + "\n\t*".join(autosklearn.metrics.CLASSIFICATION_METRICS))

    print("-------- METRICA OBTENIDA EN NUESTRO CASO -------- ")
    predictions = automl.predict(X_test)
    print("Accuracy score:", sklearn.metrics.accuracy_score(y_test, predictions))

    print("Comprobacion manual - Ejemplos de y_test cuyas dimensiones son " + str(y_test.shape[0]) )
    print(y_test.head())

    print("Comprobacion manual - Ejemplos de predictions cuyas dimensiones son " + str(predictions.shape[0]) )
    print(predictions.head())




