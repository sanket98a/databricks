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
# MAGIC * ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Use single-machine Hyperopt with a distributed training algorithm (e.g. MLlib)
# MAGIC * Use distributed Hyperopt with single-machine training algorithms (e.g. scikit-learn) with the SparkTrials class. 

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
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler

# hyperparameter tuning
from hyperopt import fmin
from hyperopt import hp
from hyperopt import space_eval
from hyperopt import STATUS_OK
from hyperopt import tpe
from hyperopt import Trials

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data prep

# COMMAND ----------

# We've created a utility function that automates the procedure of importing and preparing our
# wine data as performed in the first modules notebook
data = get_wine_data()

# convert data to Spark DataFrame
sdf = spark.createDataFrame(data)

# split into train (75%) and test (25%) data sets
train_sdf, test_sdf = sdf.randomSplit([.75, .25], seed=123)

# convert features into necessary Spark VectorAssembler for Spark MLlib training purposes
features = list(data.drop(columns='quality').columns)
vec_assembler = VectorAssembler(inputCols=features, outputCol="features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Distributed training algorithm
# MAGIC
# MAGIC The following provides an example of using single-machine hyperopt with a distributed training algorithm such as MLlib's `RandomForestClassifier`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Create search space

# COMMAND ----------

search_space = {
    'num_trees': hp.qloguniform('num_trees', 4, 7, 1),
    'max_depth': hp.quniform('max_depth', 1, 30, 1),
    'max_features': hp.loguniform('max_features', -2, -0.2),
    'max_samples': hp.choice('max_samples', [0.5, 0.75, 0.9, 1]),
    'min_samples_leaf': hp.qloguniform('min_samples_leaf', 0, 5, 1),
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Define objective function

# COMMAND ----------

# establish MLlib model object
rf = RandomForestClassifier(labelCol="quality", seed=123)
pipeline = Pipeline(stages=[vec_assembler, rf])
clf_evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="quality")


def train_model(params):    
    # set the hyperparameters that we want to tune
    estimator = pipeline.copy({
        rf.numTrees: params["num_trees"],
        rf.maxDepth: params["max_depth"], 
        rf.featureSubsetStrategy: f'{params["max_features"]}',
        rf.subsamplingRate: params["max_samples"],
        rf.minInstancesPerNode: params["min_samples_leaf"]
    })
    model = estimator.fit(train_sdf)

    preds = model.transform(test_sdf)
    auc = clf_evaluator.evaluate(preds)

    # Return -1*auc so fmin maximizes the AUC
    return {'status': STATUS_OK, 'loss': -1*auc}

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Create `Trials` object
# MAGIC
# MAGIC We create the same type of `Trials` object as we introduced in the first demo.

# COMMAND ----------

trials = Trials()

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
  trials=trials
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