# Databricks notebook source
# MAGIC
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Grid Search for Random Forests
# MAGIC
# MAGIC **Objective**: *Demonstrate the grid-search process using a validation set.*
# MAGIC
# MAGIC In this demo, we will complete a series of exercises to automate the optimization of hyperparameters.

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Prepare data
# MAGIC
# MAGIC We'll use the same data we used in this lesson's previous demo.
# MAGIC
# MAGIC As a reminder, we'll create an **`ihodb.ht_user_metrics`** table. This table will be at the user-level. 
# MAGIC
# MAGIC We'll alse be adding a new binary column **`steps_10000`** indicating whether or not the individual takes an average of at least 10,000 steps per day (`1` for yes, `0` for no).

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE ihodb.ht_user_metrics
# MAGIC USING DELTA LOCATION "/ihodb/ht-user-metrics" AS (
# MAGIC   SELECT avg(metrics.resting_heartrate) AS avg_resting_heartrate,
# MAGIC          avg(metrics.active_heartrate) AS avg_active_heartrate,
# MAGIC          avg(metrics.bmi) AS avg_bmi,
# MAGIC          avg(metrics.vo2) AS avg_vo2,
# MAGIC          avg(metrics.workout_minutes) AS avg_workout_minutes,
# MAGIC          CASE WHEN avg(metrics.steps) >= 10000 THEN 1 ELSE 0 END AS steps_10000
# MAGIC   FROM ihodb.ht_daily_metrics metrics
# MAGIC   INNER JOIN ihodb.ht_users users ON metrics.device_id = users.device_id
# MAGIC   GROUP BY metrics.device_id, users.lifestyle
# MAGIC )

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### Train-Validation-Test Split
# MAGIC
# MAGIC Our first step is to separate out our true holdout set, our test set.
# MAGIC
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> Notice the holdout set is a bit smaller this time – this is to maximize the amount of data we can use on the training set and validation set.

# COMMAND ----------

from sklearn.model_selection import train_test_split

ht_user_metrics_pd_df = spark.table("ihodb.ht_user_metrics").toPandas()

train_val_df, test_df = train_test_split(ht_user_metrics_pd_df, train_size=0.85, test_size=0.15, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC We now have two DataFrames: `train_val_df` and `test_df`. It should be noted that `train_val_df` contains the data for both the training set and the validation set – we haven't separated those yet.
# MAGIC
# MAGIC We need to perform the `train_test_split` again to separate `train_val_df` into a training set and a validation set.

# COMMAND ----------

train_df, val_df = train_test_split(train_val_df, train_size=0.7, test_size=0.3, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we have our three DataFrames:
# MAGIC
# MAGIC 1. `train_df`
# MAGIC 1. `val_df`
# MAGIC 1. `test_df`
# MAGIC
# MAGIC But keep in mind we also have `train_val_df`, which is a combination of our training set and our validation set.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyperparameter Tuning via Grid Search
# MAGIC
# MAGIC As a reminder, we're building a random forest to predict whether each user takes 10,000 steps per day.
# MAGIC
# MAGIC ### Random Forest
# MAGIC
# MAGIC We'll start by defining our random forest estimator.

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=42)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Hyperparameter Grid
# MAGIC
# MAGIC Our first step is to create a hyperparameter grid.
# MAGIC
# MAGIC We'll focus on two hyperparameters:
# MAGIC
# MAGIC 1. `max_depth` - the maximum depth of each tree
# MAGIC 2. `n_estimators` – the number of trees in the forest
# MAGIC
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> The `parameter_grid` is just a Python dictionary of values that we are predefining.

# COMMAND ----------

parameter_grid = {
  'max_depth':[2, 4, 5, 8, 10, 15, 20, 25, 30], 
  'n_estimators':[3, 5, 10, 50, 100, 150, 250, 500]
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Predefined Split
# MAGIC
# MAGIC To use our validation set in an automated process, we need to create a predefined split to pass into a our grid-search process. This is because the grid-search process takes a single DataFrame – for us this will be `train_val_df`.
# MAGIC
# MAGIC See this [note](https://scikit-learn.org/stable/modules/cross_validation.html#predefined-fold-splits-validation-sets) in the documentation for more explanation.

# COMMAND ----------

from sklearn.model_selection import PredefinedSplit

# Create list of -1s for training set row or 0s for validation set row
split_index = [-1 if row in train_df.index else 0 for row in train_val_df.index]

# Create predefined split object
predefined_split = PredefinedSplit(test_fold=split_index)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Grid Search
# MAGIC
# MAGIC We are now ready to create our grid-search object. We'll use each of the objects we've created thus far.

# COMMAND ----------

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(estimator=rfc, cv=predefined_split, param_grid=parameter_grid)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training the Models
# MAGIC
# MAGIC Now that we've created our `grid_search` object, we're ready to perform the process. `sklearn` makes this easy by implementing a familiar `fit` method to `grid_search`.

# COMMAND ----------

grid_search.fit(train_val_df.drop("steps_10000", axis=1), train_val_df["steps_10000"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optimal Hyperparameters
# MAGIC
# MAGIC If we're curious about what our optimal hyperparameters values are, we can access them pretty easily.

# COMMAND ----------

grid_search.best_params_

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC And we can also see the validation accuracy associated with these values.
# MAGIC
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> This might not be quite as high as expected because remember that it was trained on less data.

# COMMAND ----------

grid_search.best_score_

# COMMAND ----------

# MAGIC %md
# MAGIC ### Final Evaluation
# MAGIC
# MAGIC If we want to see how the final, refit model that was trained on the entirety of `train_val_df` after the optimal hyperparameters performs, we can assess it against the test set.

# COMMAND ----------

from sklearn.metrics import accuracy_score

accuracy_score(
  test_df["steps_10000"], 
  grid_search.predict(test_df.drop("steps_10000", axis=1))
)

# COMMAND ----------

# MAGIC %md
# MAGIC Great! Now it's your opportunity to try this yourself in this lesson's lab.
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>