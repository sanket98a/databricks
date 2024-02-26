# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md 
# MAGIC # Using Hyperopt with MLflow
# MAGIC
# MAGIC A big benefit of using Hyperopt is that it is nicely integrated with MLflow. Consequently, we can leverage the many benefits of MLflow within this process. 
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/icon_hint_32.png">This notebook is not designed to teach you about MLflow. For that, we recommend you check out the following resources:
# MAGIC
# MAGIC * [Mlflow.org](https://mlflow.org/)
# MAGIC * [Databricks MLflow guide](https://docs.databricks.com/applications/mlflow/)
# MAGIC * [Machine Learning in Production: MLflow and Model Deployment](https://academy.databricks.com/instructor-led-training/ml-production)
# MAGIC * [Tracking Experiements with MLflow](https://academy.databricks.com/elearning/INT-MLET-v1-SP)
# MAGIC * [Introduction to MLflow Model Registry](https://academy.databricks.com/elearning/INT-MLFLOWREG-v1-SP)
# MAGIC * [Deploying a Machine Learning Project with MLflow Projects](https://academy.databricks.com/elearning/INT-DMMS-v1-SP)
# MAGIC
# MAGIC Rather, in this notebook you'll learn how to:
# MAGIC
# MAGIC ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Setup your workflow so that MLflow will track your Hyperopt tuning experiments.

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
from hyperopt import Trials
from hyperopt.pyll import scope

# model experimentation tracking
import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data prep

# COMMAND ----------

# We've created a utility function that automates the procedure of importing and preparing our
# wine data as performed in the first module's notebook
data = get_wine_data()

# split data into train (75%) and test (25%) sets
train, test = train_test_split(data, random_state=123)
X_train = train.drop(columns="quality")
X_test = test.drop(columns="quality")
y_train = train["quality"]
y_test = test["quality"]

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Hyperopt and MLflow with `SparkTrials`
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/icon_best_32.png"> When using the `SparkTrials` class for your Hyperopt experiments, MLflow will automatically track your experiment!

# COMMAND ----------

# define search space
search_space = {
    'n_estimators': scope.int(hp.loguniform('n_estimators', 4, 7)),
    'max_depth': scope.int(hp.uniform('max_depth', 1, 100)),
    'max_features': hp.loguniform('max_features', -2, -0.2),
    'max_samples': hp.choice('max_samples', [0.5, 0.75, 0.9, 1]),
    'min_samples_leaf': scope.int(hp.loguniform('min_samples_leaf', 0, 5)),
    'random_state': 123,
}

# COMMAND ----------

# define objective function
def train_model(params):

    clf = RandomForestClassifier(**params, n_jobs=-1)
    _ = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    auc = roc_auc_score(y_test, pred)
    
    # Set the loss to -1*auc so fmin maximizes the accuracy
    return {'status': STATUS_OK, 'loss': -1*auc}

# COMMAND ----------

# define trials object
spark_trials = SparkTrials(parallelism=8)

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_32.png"> Although when using `SparkTrials` MLflow will automatically create an experiment to track the Hyperopt trials; you can also set your experiment explicitly if you desire.

# COMMAND ----------

# get user path
username = spark.sql("SELECT current_user()").first()[0].lower()

# path to MLFlow experimentation
experiment_path = f"/Users/{username}/hyperopt_example"
mlflow.set_experiment(experiment_path)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can run `fmin()` as usually. <img src="https://files.training.databricks.com/images/icon_note_32.png"> the _"Hyperopt with SparkTrials will automatically track trials in MLflow."_ message.

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

# MAGIC %md 
# MAGIC ## Hyperopt and MLFlow with `Trials`
# MAGIC
# MAGIC But what about when we use the `Trials` object rather than `SparkTrials`? Let's re-run the above procedure with `Trials`, and see if MLflow tracks our trials.

# COMMAND ----------

# define trials object
trials = Trials()

# COMMAND ----------

best_params = fmin(
  fn=train_model, 
  space=search_space, 
  algo=tpe.suggest, 
  max_evals=25,
  rstate=np.random.RandomState(123),
  trials=trials
  )

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://files.training.databricks.com/images/icon_warn_32.png"> As we found out MLflow will not automatically track Hyperopt trials when using the `Trials` object. But there may be valid reasons you need to use `Trials` (i.e. using a distributed algorith such as MLlib). So how do we use MLflow with `Trials`?
# MAGIC
# MAGIC First, we modify our Hyperopt objective function to include MLflow. Here we use `nested=True` to nest each hyperparameter trial run under a single `fmin()` run.

# COMMAND ----------

# define objective function
def train_model(params):
    
    # allows us to nest each run under the main fmin() run
    with mlflow.start_run(nested=True):
    
        # we can use autolog or any other MLFlow logging functions to record to our experiment
        mlflow.autolog(silent=True)
    
        # the rest follows are normal objective function procedure
        clf = RandomForestClassifier(**params, n_jobs=-1)
        _ = clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        auc = roc_auc_score(y_test, pred)
        mlflow.log_metric('test_auc', auc)
    
        # Set the loss to -1*auc so fmin maximizes the accuracy
        return {'status': STATUS_OK, 'loss': -1*auc}

# COMMAND ----------

# MAGIC %md
# MAGIC Now we just run `fmin` within an `mlflow.start_run`call:

# COMMAND ----------

# reset our trials object
trials = Trials()

# recorded our nested hyperparameter trials under a single "trial2" run
with mlflow.start_run(run_name='trial2'):  
    best_params = fmin(
        fn=train_model,
        space=search_space, 
        algo=tpe.suggest, 
        max_evals=25,
        rstate=np.random.RandomState(123),
        trials=trials
    )

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_32.png"> One thing to note is that the above code does not record the overall best trial loss for the parent trial2 run. We can add a few more lines of code to:
# MAGIC
# MAGIC 1. extract and log best parameters found from our Hyperopt trials,
# MAGIC 2. fit final model with best parameters,
# MAGIC 3. score the final model and record the best trial loss.

# COMMAND ----------

# reset our trials object
trials = Trials()

# recorded our nested hyperparameter trials under a single "trial3" run
with mlflow.start_run(run_name='trial3'):  
    best_params = fmin(
        fn=train_model,
        space=search_space, 
        algo=tpe.suggest, 
        max_evals=25,
        rstate=np.random.RandomState(123),
        trials=trials
    )
    
    # extract and log best parameters found
    final_params = space_eval(search_space, best_params)
    mlflow.log_params(final_params)
    
    # fit final model with best parameters
    clf = RandomForestClassifier(**final_params, n_jobs=-1)
    _ = clf.fit(X_train, y_train)
    
    # score model on test data and record loss
    pred = clf.predict(X_test)
    auc = roc_auc_score(y_test, pred)
    mlflow.log_metric('best_trial_loss', auc)
    

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