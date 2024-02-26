# Databricks notebook source
# MAGIC
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # K-fold Cross-Validation with Random Forest
# MAGIC
# MAGIC **Objective**: *Demonstrate the the use of K-fold cross-validation.*
# MAGIC
# MAGIC In this demo, we will complete a series of exercises to identify optimal hyperparameters using cross-validation and grid-search.

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Prepare data
# MAGIC
# MAGIC We'll create an **`ihodb.ht_user_metrics`** table. This table will be at the user-level. 
# MAGIC
# MAGIC We'll alse be adding a new binary column **`steps_10000`** indicating whether or not the individual takes an average of at least 10,000 steps per day (`1` for yes, `0` for no).
# MAGIC
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> Note that we're using fewer features than the previous lab.

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
# MAGIC ### Train-Test Split
# MAGIC
# MAGIC When we perform cross-validation, remember that it's still important to separate out the cross-validation set from the test (or holdout) set.
# MAGIC
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> We are using 80 percent of data for cross-validation and 20 percent of data for test. This is because we no longer have to split the non-test data between training and validation sets.

# COMMAND ----------

from sklearn.model_selection import train_test_split

ht_user_metrics_pd_df = spark.table("ihodb.ht_user_metrics").toPandas()

cross_val_df, test_df = train_test_split(ht_user_metrics_pd_df, train_size=0.80, test_size=0.20, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC We now have two DataFrames: `cross_val_df` and `test_df`. It should be noted that `train_val_df` contains all of the folds of our cross-validation data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyperparameter Tuning via Grid Search with Cross-Validation
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

# MAGIC %md
# MAGIC ### Hyperparameter Grid
# MAGIC
# MAGIC Just like in the last lesson, our first step is to create a hyperparameter grid.
# MAGIC
# MAGIC We'll focus on two hyperparameters:
# MAGIC
# MAGIC 1. `max_depth` - the maximum depth of each tree
# MAGIC 2. `n_estimators` – the number of trees in the forest

# COMMAND ----------

parameter_grid = {
  'max_depth':[2, 4, 5, 8, 10, 15, 20, 25], 
  'n_estimators':[3, 5, 10, 50, 100]
}

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Cross-Validated Grid Search
# MAGIC
# MAGIC We are now ready to create our grid-search object. We'll use each of the objects we've created thus far.
# MAGIC
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> Instead of passing a `PredefinedSplit` object to the `cv` parameter, we're simply passing the number of folds.

# COMMAND ----------

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(estimator=rfc, cv=3, param_grid=parameter_grid)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training the Models
# MAGIC
# MAGIC Now that we've created our `grid_search` object, we're ready to perform the process.
# MAGIC
# MAGIC This is the same process at the previous lesson.

# COMMAND ----------

grid_search.fit(cross_val_df.drop("steps_10000", axis=1), cross_val_df["steps_10000"])

# COMMAND ----------

# MAGIC %md
# MAGIC **Question:** How many models are we training right now?
# MAGIC
# MAGIC *Number of Unique Hyperparameter Combinations* x *Number of Folds* + 1

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cross-validated Results
# MAGIC
# MAGIC If you want to examine the results for each individual fold, you can use `grid_search`'s `cv_results_` attribute.
# MAGIC
# MAGIC Note that each row of the DataFrame corresponds to a unique set of hyperparameters.

# COMMAND ----------

import pandas as pd
pd.DataFrame(grid_search.cv_results_).head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optimal Hyperparameters
# MAGIC
# MAGIC If you don't want to dig through the above DataFrame to determine your optimal hyperparameters, you can still access them using `best_params_`.

# COMMAND ----------

grid_search.best_params_

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC And we can also see the average accuracy associated with these hyperparameter values.
# MAGIC
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> This is a bit better than we saw with the train-validation-test split.

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
# MAGIC Great! We have one more lecture video before we finish off the lesson with lab.
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>