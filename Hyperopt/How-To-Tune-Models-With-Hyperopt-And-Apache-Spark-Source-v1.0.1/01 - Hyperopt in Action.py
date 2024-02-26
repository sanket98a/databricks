# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Hyperopt in Action
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC
# MAGIC - Learn about the four important features required to run your first SMBO hyperparameter search with hyperopt
# MAGIC - Use hyperopt to find the optimal parameters for a random forest classifier model

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC First, you’ll want to make sure you have a proper cluster started. In this case we’re going to use a Databricks Runtime of 9.1 but we want to make sure we are using the Machine Learning version. This particular cluster uses Standard mode, has 4 cores and 8 workers.
# MAGIC
# MAGIC ### Classroom setup
# MAGIC
# MAGIC In each notebook you will see the following code cell, which we use to do some basic setup for your environment such as transfer data from our central location to your environment. This is the first cell you will need to run prior to any code that follows.

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

#%fs ls "dbfs:/databricks-datasets/"

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Libraries

# COMMAND ----------

# MAGIC %pip install hyperopt

# COMMAND ----------

# helper packages
import pandas as pd
import math
import numpy as np
import time

# modeling
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# hyperparameter tuning
import hyperopt

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data
# MAGIC
# MAGIC For simplicity we will use the well known [wine quality dataset](https://archive-beta.ics.uci.edu/ml/datasets/wine+quality)  (Cortez et al., 2009) provided by the [UCI Machine Learning Repository](https://archive-beta.ics.uci.edu/). Our objective in this modeling task is to use wine characteristics in order to predict the quality of the wine. In this example we are approaching this as a classification problem with the objective of predicting if a wine is high quality ( \\( quality \\geq 7 \\) ) or low quality ( \\( quality < 7 \\) ). Here, we use a utility function included in our classroom setup that simplifies our data prep by performing the following tasks:
# MAGIC
# MAGIC 1. Read in the data
# MAGIC 2. Create indicator column for red vs. white wine
# MAGIC 3. Combine the red and white wine datasets
# MAGIC 4. Clean up column names
# MAGIC 5. Convert wine quality column to a binary response variable

# COMMAND ----------

data = get_wine_data()

data.head(10)

# COMMAND ----------

data.info()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Next, we'll split our data into train and test sets using the default 75% (train) 25% (test) ratio.

# COMMAND ----------

# split data into train (75%) and test (25%) sets
train, test = train_test_split(data, random_state=123)
X_train = train.drop(columns="quality")
X_test = test.drop(columns="quality")
y_train = train["quality"]
y_test = test["quality"]

X_train.head()

# COMMAND ----------

y_train.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Features of Hyperopt
# MAGIC
# MAGIC Hyperopt contains 4 important features you need to know in order to run your first SMBO hyperparameter search:
# MAGIC
# MAGIC 1. Search space
# MAGIC 2. Objective function
# MAGIC 3. Trial object
# MAGIC 4. Optimization function

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Search Space
# MAGIC
# MAGIC Hyperopt provides various functions via `hyperopt.hp` to specify ranges for input parameters, these are stochastic search spaces. The most common options for a search space to choose are :
# MAGIC
# MAGIC * `hp.choice(label, options)` — This can be used for categorical parameters, it returns one of the options, which should be a list or tuple. Example: `hp.choice('subsample', [0.5, 0.75, 0.9, 1])`
# MAGIC * `hp.normal(label, mu, sigma)` — This returns a real value that’s normally-distributed with mean mu and standard deviation sigma. Example: `hp.normal('gamma', 10, 2)`
# MAGIC * `hp.lognormal(label, mu, sigma)` — This returns a value drawn according to exp(normal(mu, sigma))
# MAGIC * `hp.uniform(label, low, high)` — It returns a value uniformly between low and high. Example: `hp.uniform('max_leaf_nodes', 1, 10)`
# MAGIC * `hp.loguniform(label, low, high)` - Returns a value drawn according to exp(uniform(low, high)) so that the logarithm of the return value is uniformly distributed. Example: `hp.loguniform('learning_rate', -7, 0)`
# MAGIC
# MAGIC Other option you can use are:
# MAGIC
# MAGIC * `hp.randint(label, upper)` — Returns a random integer in the range (0, upper).
# MAGIC * `hp.qnormal(label, mu, sigma, q)` — Returns a value drawn according to round(normal(mu, sigma) / q) * q
# MAGIC * `hp.qlognormal(label, mu, sigma, q)` — Returns a value drawn according to round(exp(normal(mu, sigma)) / q) * q
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_32.png"> You can learn more search space options [here](http://hyperopt.github.io/hyperopt/getting-started/search_spaces/#parameter-expressions).
# MAGIC
# MAGIC The following is an example of creating a hyperparameter distribution to be drawn from using `loguniform()`. You can see the output is not an actual value but, rather, a class that will be used to extract future values.

# COMMAND ----------

example_hyper = hyperopt.hp.loguniform('my_label', -9, -1) # range e^{low} to e^{high}
example_hyper

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://files.training.databricks.com/images/icon_warn_32.png"> It's important to read the docs so you understand the distribution parameters you are applying. For example, the `loguniform()` low and high values are actually `exp(low)` & `exp(high)`. Consequently, the above example...

# COMMAND ----------

f'equates to searching log space of {math.exp(-9)} to {math.exp(-1)}'

# COMMAND ----------

# MAGIC %md
# MAGIC We can use `hyperopt.pyll.stochastic.sample()` to sample from the distribution. The below plots 1,000 samples from our `loguniform` distribution.

# COMMAND ----------

sampled_values = [hyperopt.pyll.stochastic.sample(example_hyper) for i in range(1000)]
pd.Series(sampled_values).hist(bins=30, log=True)

# COMMAND ----------

# MAGIC %md
# MAGIC We can use these distributional search spaces together to form our overall search space. When establishing the range of values, choose bounds that are extreme and let hyperopt learn what values aren’t working well. For example, if a regularization parameter is typically between 1 and 10, try values from 0 to 100. The range should include the default value, certainly. At worst, it may spend time trying extreme values that do not work well at all, but it should learn and stop wasting trials on bad value.
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/icon_best_32.png"> A best practice strategy for a hyperopt workflow is as follows:
# MAGIC
# MAGIC 1. Choose what hyperparameters are reasonable to optimize
# MAGIC 2. Define broad ranges for each of the hyperparameters (including the default where applicable)
# MAGIC 3. Run a small number of trials
# MAGIC 4. Observe the results in an MLflow parallel coordinate plot and select the runs with lowest loss
# MAGIC 5. Move the range towards those higher/lower values when the best runs’ hyperparameter values are pushed against one end of a range
# MAGIC 6. Determine whether certain hyperparameter values cause fitting to take a long time (and avoid those values)
# MAGIC 7. Re-run with more trials
# MAGIC 8. Repeat until the best runs are comfortably within the given search bounds and none are taking excessive time

# COMMAND ----------

search_space = {
    'n_estimators': hyperopt.pyll.scope.int(hyperopt.hp.qloguniform('n_estimators', 4, 7, 1)),
    'max_depth': hyperopt.hp.quniform('max_depth', 1, 100, 1),
    'max_features': hyperopt.hp.loguniform('max_features', -2, -0.2),
    'max_samples': hyperopt.hp.choice('max_samples', [0.5, 0.75, 0.9, 1]),
    'min_samples_leaf': hyperopt.pyll.scope.int(hyperopt.hp.qloguniform('min_samples_leaf', 0, 5, 1)),
    'random_state': int(10),
}

# COMMAND ----------

# an example sample from our overall search space
hyperopt.pyll.stochastic.sample(search_space)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Objective function
# MAGIC
# MAGIC When using hyperopt, we need to create an objective function that will be optimized by Hyperopt's Bayesian optimizer. The function definition typically only requires one parameter (`params`), which Hyperopt will use to pass a set of hyperparameter values. So, given a set of hyperparameter values that hyperopt chooses, the function trains our given model and computes the loss for the model built with those hyperparameters. 
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/icon_best_32.png">It's common for Hyperopt objective functions to return a dictionary with:
# MAGIC
# MAGIC * __status__: One of the keys from `hyperopt.STATUS_STRINGS` to signal successful vs. failed completion. Most common is `STATUS_OK`.
# MAGIC * __loss__: The loss score to be optimized. Note that Hyperopt will always try to minimize the loss so if you choose a loss where the objective is to maximize (i.e. \\(R^2\\), AUC, accuracy) then you will need to negate this value.

# COMMAND ----------

def train_model(params):
    clf = RandomForestClassifier(**params, n_jobs=-1)
    _ = clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    
    # Return -1*acc so fmin maximizes the accuracy
    return {'status': hyperopt.STATUS_OK, 'loss': -1*acc}

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Trial Object
# MAGIC
# MAGIC The Trials object is not necessary but allows you to record information from each hyperparameter run (i.e. loss, hyperparameter values, best results). This allows you to access results after running search space optimization. 

# COMMAND ----------

trials = hyperopt.Trials()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Optimization function
# MAGIC
# MAGIC To execute the search we use `hyperopt.fmin()` and supply it our model training (objective) function along with the hyperparameter search space. `fmin` can use different algorithms to search across the hyperparameter search space (i.e. random, Bayesian); however, we suggest using the ***Tree of Parzen Estimators*** (`tpe.suggest`) which will perform a Bayesian SMBO hyperparameter search.  
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_32.png"> It is smart to run an initial small number of trials to find a range of hyperparameter values that appear to perform well and then refine the search space. Note that 'small' will be relative to compute time and size of the search space. There a few ways to limit the search:
# MAGIC
# MAGIC * `max_evals`: maximum number of trials to run. Example `max_evals=25`.
# MAGIC * `loss_threshold`: stop the grid search once we've reached loss threshold. Example `loss_threshold=-0.92` will search until an AUC of 0.92 or higher is obtained.
# MAGIC * `timeout`: stop after specified number of seconds. Example: `timeout=60*10` will stop after 10 minutes.
# MAGIC
# MAGIC The following example runs 25 trials for the initial search.

# COMMAND ----------

best_params = hyperopt.fmin(
         fn=train_model,
         space=search_space,
         algo=hyperopt.tpe.suggest,
         max_evals=25,
         trials=trials)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Results
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/icon_warn_32.png"> The output of `fmin` provides the hyperparameters with the best result; however, it will return results from `hp.choice` as an index! Also, notice that `max_depth` is not in integer form.

# COMMAND ----------

best_params

# COMMAND ----------

trials.results

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://files.training.databricks.com/images/icon_best_32.png"> So it is better to use `space_eval()` which will return the actual hyperparameter values used during the optimal run:

# COMMAND ----------

hyperopt.space_eval(search_space, best_params)

# COMMAND ----------

# MAGIC %md
# MAGIC Our trials object will contain results from  our runs. Some useful information it contains includes:
# MAGIC
# MAGIC * `trials.results`
# MAGIC * `trials.losses()`
# MAGIC * `trials.statuses()`

# COMMAND ----------

trials.best_trial

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