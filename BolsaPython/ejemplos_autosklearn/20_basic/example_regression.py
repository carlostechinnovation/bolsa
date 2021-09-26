# -*- encoding: utf-8 -*-

import autosklearn.regression
##### Regression
import sklearn.datasets
import sklearn.metrics

print("-------- Limpieza -------- ")
#shutil.rmtree("/tmp/autosklearn_regression_example_tmp")
#shutil.rmtree("/tmp/autosklearn_regression_example_out")

print("-------- Data Loading -------- ")
X, y = sklearn.datasets.load_boston(return_X_y=True)
#X, y = sklearn.datasets.load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)

print("-------- Build and fit a regressor -------- ")
automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=120,    per_run_time_limit=30,
    tmp_folder='/tmp/autosklearn_regression_example_tmp',
    output_folder='/tmp/autosklearn_regression_example_out',
include_estimators=None,
    )
automl.fit(X_train, y_train, dataset_name='boston')

print("-------- Print the final ensemble constructed by auto-sklearn -------- ")
print(automl.show_models())

print("-------- SCORE OF THE ENSEMBLE -------- ")
print("Available REGRESSION autosklearn.metrics.*:")
print("\t*" + "\n\t*".join(autosklearn.metrics.REGRESSION_METRICS))

predictions = automl.predict(X_test)
print("R2 score:", sklearn.metrics.r2_score(y_test, predictions))

