# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Distributing Hyperopt Procedures
# MAGIC
# MAGIC In the machine learning workflow, Hyperopt can be used to distribute/parallelize the hyperparameter optimization process with more advanced optimization strategies than are available in other libraries.
# MAGIC
# MAGIC There are two ways to scale Hyperopt with Apache Spark:
# MAGIC
# MAGIC * Use single-machine Hyperopt with a distributed training algorithm (e.g. MLlib)
# MAGIC * ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Use distributed Hyperopt with single-machine training algorithms (e.g. scikit-learn) with the SparkTrials class. 

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Prerequisites
# MAGIC
# MAGIC ### Classroom setup

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Libraries

# COMMAND ----------

# helper packages
import pandas as pd
import numpy as np

# modeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# hyperparameter tuning
from hyperopt import fmin
from hyperopt import hp
from hyperopt import space_eval
from hyperopt import SparkTrials
from hyperopt import STATUS_OK
from hyperopt import tpe
from hyperopt.pyll import scope

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data prep

# COMMAND ----------

# We've created a utility function that automates the procedure of importing and preparing our
# wine data as performed in the first modules notebook
data = get_wine_data()

# split data into train (75%) and test (25%) sets
train, test = train_test_split(data, random_state=123)
X_train = train.drop(columns="quality")
X_test = test.drop(columns="quality")
y_train = train["quality"]
y_test = test["quality"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Distributed hyperopt
# MAGIC
# MAGIC The following provides an example of distributing hyperopt with single-machine training algorithms (e.g. scikit-learn) with the `SparkTrials` class.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Create search space

# COMMAND ----------

search_space = {
    'n_estimators': scope.int(hp.loguniform('n_estimators', 4, 7)),
    'max_depth': scope.int(hp.uniform('max_depth', 1, 100)),
    'max_features': hp.loguniform('max_features', -2, -0.2),
    'max_samples': hp.choice('max_samples', [0.5, 0.75, 0.9, 1]),
    'min_samples_leaf': scope.int(hp.loguniform('min_samples_leaf', 0, 5)),
    'random_state': 123,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Define objective function

# COMMAND ----------

def train_model(params):
    clf = RandomForestClassifier(**params, n_jobs=-1)
    _ = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    auc = roc_auc_score(y_test, pred)
    
    # Set the loss to -1*auc so fmin maximizes the accuracy
    return {'status': STATUS_OK, 'loss': -1*auc}

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Create `SparkTrials` object
# MAGIC
# MAGIC Here, we introduce a new type of trials object called `SparkTrials`, which allows you to distribute a Hyperopt run without making other changes to your Hyperopt code.
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/icon_warn_32.png"> `SparkTrials` is designed to parallelize computations for single-machine ML models such as Scikit-learn. For models created with distributed ML algorithms such as MLlib or Horovod, do not use `SparkTrials`.
# MAGIC
# MAGIC `SparkTrials` takes two optional arguments:
# MAGIC
# MAGIC * `parallelism`: Maximum number of trials to evaluate concurrently. A higher number lets you scale-out testing of more hyperparameter settings. Because Hyperopt proposes new trials based on past results, there is a trade-off between parallelism and adaptivity. For a fixed `max_evals`, greater parallelism speeds up calculations, but lower parallelism may lead to better results since each iteration has access to more past results.
# MAGIC
# MAGIC   Default: Number of Spark executors available. Maximum: 128. If the value is greater than the number of concurrent tasks allowed by the cluster configuration, SparkTrials reduces parallelism to this value.
# MAGIC
# MAGIC * `timeout`: Maximum number of seconds an `fmin()` call can take. When this number is exceeded, all runs are terminated and `fmin()` exits. Information about completed runs is saved.

# COMMAND ----------

spark_trials = SparkTrials(parallelism=8)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Run optimization function

# COMMAND ----------

best_params = fmin(
  fn=train_model, 
  space=search_space, 
  algo=tpe.suggest, 
  max_evals=25,
  rstate=np.random.RandomState(123),
  trials=spark_trials
  )

# COMMAND ----------

space_eval(search_space, best_params)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classroom Celanup
# MAGIC Remove any temp files and databases created by this lesson

# COMMAND ----------

classroom_cleanup()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>