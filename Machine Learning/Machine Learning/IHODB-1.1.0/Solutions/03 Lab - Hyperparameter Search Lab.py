# Databricks notebook source
# MAGIC
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Hyperparameter Search Lab
# MAGIC
# MAGIC **Objective**: *Apply grid-search hyperparameter optimization to improve the performance of a model.*
# MAGIC
# MAGIC In this lab, you will apply what you've learned in this lesson.

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Exercise 1
# MAGIC
# MAGIC In this exercise, you will create an enhanced user-level table to try to better predict whether or not each user takes at least 10,000 steps in a day.
# MAGIC
# MAGIC Fill in the blanks in the below cell to create the `ihodb.ht_user_metrics_hs_lab` table.
# MAGIC
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** Refer back to previous demos on how to create the `steps_10000` column.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ANSWER
# MAGIC CREATE OR REPLACE TABLE ihodb.ht_user_metrics_hs_lab
# MAGIC USING DELTA LOCATION "/ihodb/ht-user-metrics-hs-lab" AS (
# MAGIC   SELECT min(resting_heartrate) AS min_resting_heartrate,
# MAGIC          avg(resting_heartrate) AS avg_resting_heartrate,
# MAGIC          max(resting_heartrate) AS max_resting_heartrate,
# MAGIC          max(resting_heartrate) - min(resting_heartrate) AS resting_heartrate_change,
# MAGIC          min(active_heartrate) AS min_active_heartrate,
# MAGIC          avg(active_heartrate) AS avg_active_heartrate,
# MAGIC          max(active_heartrate) AS max_active_heartrate,
# MAGIC          max(active_heartrate) - min(active_heartrate) AS active_heartrate_change,
# MAGIC          min(bmi) AS min_bmi,
# MAGIC          avg(bmi) AS avg_bmi,
# MAGIC          max(bmi) AS max_bmi,
# MAGIC          max(bmi) - min(bmi) AS bmi_change,
# MAGIC          min(vo2) AS min_vo2,
# MAGIC          avg(vo2) AS avg_vo2,
# MAGIC          max(vo2) AS max_vo2,
# MAGIC          max(vo2) - min(vo2) AS vo2_change,
# MAGIC          min(workout_minutes) AS min_workout_minutes,
# MAGIC          avg(workout_minutes) AS avg_workout_minutes,
# MAGIC          max(workout_minutes) AS max_workout_minutes,
# MAGIC          max(workout_minutes) - min(workout_minutes) AS workout_minutes_change,
# MAGIC          CASE WHEN avg(steps) >= 10000 THEN 1 ELSE 0 END AS steps_10000
# MAGIC   FROM ihodb.ht_daily_metrics
# MAGIC   GROUP BY device_id
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC **Question:** How many users in `ihodb.ht_user_metrics_hs_lab` take, on average, 10,000 steps per day?

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT steps_10000, count(*) FROM ihodb.ht_user_metrics_hs_lab GROUP BY steps_10000

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Exercise 2
# MAGIC
# MAGIC In this exercise, you will split your data into a training set (`train_df`), validation set (`val_df`), and test set (`test_df`).
# MAGIC
# MAGIC Fill in the blanks below to split your data.
# MAGIC
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** Refer to the previous demo for guidance.

# COMMAND ----------

# ANSWER
from sklearn.model_selection import train_test_split

ht_user_metrics_pd_df = spark.table("ihodb.ht_user_metrics_hs_lab").toPandas()

train_val_df, test_df = train_test_split(ht_user_metrics_pd_df, train_size=0.85, test_size=0.15, random_state=42)
train_df, val_df = train_test_split(train_val_df, train_size=0.7, test_size=0.3, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC **Question:** How many rows are in the `val_df` DataFrame?

# COMMAND ----------

val_df.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 3
# MAGIC
# MAGIC In this exercise, you will prepare your random forest classifier.
# MAGIC
# MAGIC Fill in the blanks below to complete the task.

# COMMAND ----------

# ANSWER
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 4
# MAGIC
# MAGIC In this exercise, you will create a hyperparameter grid to use during the grid search process.
# MAGIC
# MAGIC Use the following hyperparameter values:
# MAGIC
# MAGIC 1. `max_depth`: 2, 3, 5, 8, 10, 15
# MAGIC 1. `n_estimators`: 5, 10, 25, 50, 100, 250
# MAGIC 1. `min_samples_split`: 2, 3, 4
# MAGIC 1. `min_impurity_decrease`: 0.0, 0.01, 0.05
# MAGIC
# MAGIC Fill in the blanks below to create the grid.

# COMMAND ----------

# ANSWER
parameter_grid = {
  "max_depth": [2, 3, 5, 8, 10, 15],
  "n_estimators": [5, 10, 25, 50, 100, 250],
  "min_samples_split": [2, 3, 4],
  "min_impurity_decrease": [0.0, 0.01, 0.05]
}

# COMMAND ----------

# MAGIC %md
# MAGIC **Question**: How many total unique combinations of hyperparameters are there in `parameter_grid`? 

# COMMAND ----------

len(parameter_grid["max_depth"]) * len(parameter_grid["n_estimators"]) * len(parameter_grid["min_samples_split"]) * len(parameter_grid["min_impurity_decrease"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 5
# MAGIC
# MAGIC In this exercise, you will create a predefined split for your training set and your validation set.
# MAGIC
# MAGIC Fill in the blanks below to create the PredefinedSplit.

# COMMAND ----------

# ANSWER
from sklearn.model_selection import PredefinedSplit

# Create list of -1s for training set row or 0s for validation set row
split_index = [-1 if row in train_df.index else 0 for row in train_val_df.index]

# Create predefined split object
predefined_split = PredefinedSplit(test_fold=split_index)

# COMMAND ----------

# MAGIC %md
# MAGIC **Question**: How many 0s are there in `split_index`?

# COMMAND ----------

split_index.count(0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 6
# MAGIC
# MAGIC In this exercise, you will create the grid search object that you will use to optimize your hyperparameter values.
# MAGIC
# MAGIC Fill in the blanks below to create the object.

# COMMAND ----------

# ANSWER
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(estimator=rfc, cv=predefined_split, param_grid=parameter_grid)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 7
# MAGIC
# MAGIC In this exercise, you will fit the grid search process.
# MAGIC
# MAGIC Fill in the blanks below to perform the grid search process.

# COMMAND ----------

# ANSWER
grid_search.fit(train_val_df.drop("steps_10000", axis=1), train_val_df["steps_10000"])

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC **Question**: How many unique models are being trained by the grid search process?
# MAGIC
# MAGIC * 324
# MAGIC * 765
# MAGIC * 325
# MAGIC * 766
# MAGIC
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** Consider the number of unique feature combinations and the final retraining of the model on the training *and* validation sets.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Exercise 8
# MAGIC
# MAGIC In this exercise, you will identify the optimal hyperparameter values.
# MAGIC
# MAGIC Fill in the blanks below to indentify the optimal hyperparameter values.

# COMMAND ----------

# ANSWER
grid_search.best_params_

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Question:** What is the optimal hyperparameter value for `min_samples_split` according to the grid search process?

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Exercise 9
# MAGIC
# MAGIC In this exercise, you will identify the validation accuracy that was achieved for the optimal hyperparameter values when trained on the training set.
# MAGIC
# MAGIC Fill in the blanks below to identify the validation accuracy.

# COMMAND ----------

# ANSWER
grid_search.best_score_

# COMMAND ----------

# MAGIC %md
# MAGIC **Question:** What is the best validation set accuracy?

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Exercise 10
# MAGIC
# MAGIC In this exercise, you will identify the test accuracy achieved by the final, refit model.
# MAGIC
# MAGIC Fill in the blanks below to identify the test accuracy.

# COMMAND ----------

# ANSWER
from sklearn.metrics import accuracy_score

accuracy_score(
  test_df["steps_10000"], 
  grid_search.predict(test_df.drop("steps_10000", axis=1))
)

# COMMAND ----------

# MAGIC %md
# MAGIC **Question:** What is the test set accuracy?

# COMMAND ----------

# MAGIC %md
# MAGIC Congrats! That concludes our lab! Move on to the next demo to learn about cross-validation.
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>