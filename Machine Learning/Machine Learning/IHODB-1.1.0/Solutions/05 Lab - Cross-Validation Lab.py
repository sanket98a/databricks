# Databricks notebook source
# MAGIC
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Cross-Validation Lab
# MAGIC
# MAGIC **Objective**: *Assess your ability to apply cross-validated hyperparameter tuning to a model.*
# MAGIC
# MAGIC In this lab, you will apply what you've learned in this lesson.

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Exercise 1
# MAGIC
# MAGIC In this exercise, you will create an enhanced user-level table to try to better predict whether or not each user takes at least *8,000* steps in a day. For this exercise, assume we only have access to heart rate information.
# MAGIC
# MAGIC Fill in the blanks in the below cell to create the `ihodb.ht_user_metrics_cv_lab` table.
# MAGIC
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> Note that this lab is focused on predicting whether users take 8,000 steps per day rather than 10,000 steps per day.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ANSWER
# MAGIC CREATE OR REPLACE TABLE ihodb.ht_user_metrics_cv_lab
# MAGIC USING DELTA LOCATION "/ihodb/ht-user-metrics-cv-lab" AS (
# MAGIC   SELECT min(resting_heartrate) AS min_resting_heartrate,
# MAGIC          avg(resting_heartrate) AS avg_resting_heartrate,
# MAGIC          max(resting_heartrate) AS max_resting_heartrate,
# MAGIC          max(resting_heartrate) - min(resting_heartrate) AS resting_heartrate_change,
# MAGIC          min(active_heartrate) AS min_active_heartrate,
# MAGIC          avg(active_heartrate) AS avg_active_heartrate,
# MAGIC          max(active_heartrate) AS max_active_heartrate,
# MAGIC          max(active_heartrate) - min(active_heartrate) AS active_heartrate_change,
# MAGIC          CASE WHEN avg(steps) > 8000 THEN 1 ELSE 0 END AS steps_8000
# MAGIC   FROM ihodb.ht_daily_metrics
# MAGIC   GROUP BY device_id
# MAGIC )

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC **Question:** How many users in `ihodb.ht_user_metrics_cv_lab` take, on average, 8,000 steps per day?
# MAGIC
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** Refer back to the previous lab for guidance on how to answer this question.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Exercise 2
# MAGIC
# MAGIC In this exercise, you will split your data into a cross-validation set (`cross_val_df`) and test set (`test_df`).
# MAGIC
# MAGIC Fill in the blanks below to split your data.
# MAGIC
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** Refer to the previous demo for guidance.

# COMMAND ----------

# ANSWER
from sklearn.model_selection import train_test_split

ht_user_metrics_pd_df = spark.table("ihodb.ht_user_metrics_cv_lab").toPandas()

cross_val_df, test_df = train_test_split(ht_user_metrics_pd_df, train_size=0.80, test_size=0.20, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC **Question:** How many rows are in the `cross_val_df` DataFrame?
# MAGIC
# MAGIC Fill in the blanks below to answer the question.

# COMMAND ----------

# ANSWER
cross_val_df.shape

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
# MAGIC 1. `max_depth`: 5, 8, 20
# MAGIC 1. `n_estimators`: 25, 50, 100
# MAGIC 1. `min_samples_split`: 2, 4
# MAGIC 1. `max_features`: 3, 4
# MAGIC 1. `max_samples`: 0.6, 0.8
# MAGIC
# MAGIC Fill in the blanks below to create the grid.

# COMMAND ----------

# ANSWER
parameter_grid = {
  "max_depth": [5, 8, 20],
  "n_estimators": [25, 50, 100],
  "min_samples_split": [2, 4],
  "max_features": [3, 4],
  "max_samples": [0.6, 0.8]
}

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC **Question**: How many total unique combinations of hyperparameters are there in `parameter_grid`?
# MAGIC
# MAGIC Use the below empty cell to determine the answer to the above question.
# MAGIC
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** Refer to the previous lesson's lab for guidance.

# COMMAND ----------

# ANSWER
len(parameter_grid["max_depth"]) * len(parameter_grid["n_estimators"]) * len(parameter_grid["min_samples_split"]) * len(parameter_grid["max_features"]) * len(parameter_grid["max_samples"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 5
# MAGIC
# MAGIC In this exercise, you will create the cross-validated grid-search object that you will use to optimize your hyperparameter values while using cross-validation.
# MAGIC
# MAGIC Fill in the blanks below to create the object.
# MAGIC
# MAGIC :NOTE: Please use 4-fold cross-validation.

# COMMAND ----------

# ANSWER
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(estimator=rfc, cv=4, param_grid=parameter_grid)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 6
# MAGIC
# MAGIC In this exercise, you will fit the grid search process.
# MAGIC
# MAGIC Fill in the blanks below to perform the grid search process.

# COMMAND ----------

# ANSWER
grid_search.fit(cross_val_df.drop("steps_8000", axis=1), cross_val_df["steps_8000"])

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC **Question**: How many unique models are being trained by the cross-validated grid search process?
# MAGIC
# MAGIC * 4
# MAGIC * 289
# MAGIC * 21
# MAGIC * 288
# MAGIC
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** Consider the number of unique feature combinations, the number of cross-validation folds and the final retraining of the model on the entire cross-validation set.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Exercise 7
# MAGIC
# MAGIC In this exercise, you will return a Pandas DataFrame of the `grid_search` results.
# MAGIC
# MAGIC Fill in the blanks below to return the DataFrame.

# COMMAND ----------

# ANSWER
import pandas as pd
pd.DataFrame(grid_search.cv_results_)

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
# MAGIC **Question:** What is the optimal hyperparameter value for `max_depth` according to the cross-validated grid search process?

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Exercise 9
# MAGIC
# MAGIC In this exercise, you will identify the test accuracy achieved by the final, refit model.
# MAGIC
# MAGIC Fill in the blanks below to identify the test accuracy.

# COMMAND ----------

# ANSWER
from sklearn.metrics import accuracy_score

accuracy_score(
  test_df["steps_8000"], 
  grid_search.predict(test_df.drop("steps_8000", axis=1))
)

# COMMAND ----------

# MAGIC %md
# MAGIC **Question:** What is the test set accuracy?

# COMMAND ----------

# MAGIC %md
# MAGIC Congratulations! That concludes our lesson on cross-validated hyperparameter optimization and our course!
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>