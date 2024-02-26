# Databricks notebook source
# MAGIC
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Hyperparameters in Tree-based Models
# MAGIC
# MAGIC **Objective**: *Demonstrate the manual process of changing hyperparameters and testing values.*
# MAGIC
# MAGIC In this demo, we will complete a series of exercises to explore the use of hyperparameters.

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Prepare data
# MAGIC
# MAGIC In this demo, we'll create an **`ihodb.ht_user_metrics`** table. This table will be at the user-level. 
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

# MAGIC %md
# MAGIC We can display our new table and confirm our **`steps_10000`** column.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM ihodb.ht_user_metrics LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Train-test Split
# MAGIC
# MAGIC Remember that we need to split our training data and our test data so we can determine whether or not our models generalize well.

# COMMAND ----------

from sklearn.model_selection import train_test_split

ht_user_metrics_pd_df = spark.table("ihodb.ht_user_metrics").toPandas()

train_df, test_df = train_test_split(ht_user_metrics_pd_df, train_size=0.8, test_size=0.2, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Random Forest
# MAGIC
# MAGIC In this demo, we'll try to build a random forest to predict whether each user takes 10,000 steps per day.
# MAGIC
# MAGIC ### Hyperparameters
# MAGIC
# MAGIC We'll focus on two hyperparameters:
# MAGIC
# MAGIC 1. `max_depth` - the maximum depth of each tree
# MAGIC 2. `n_estimators` – the number of trees in the forest

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's take a look at our current values for each of these hyperparameters.

# COMMAND ----------

rfc.get_params()

# COMMAND ----------

# MAGIC %md
# MAGIC These aren't the hyperparameters we want. Let's change them.

# COMMAND ----------

rfc.set_params(max_depth=2, n_estimators=3)

# COMMAND ----------

# MAGIC %md
# MAGIC And we can verify that we changed the hyperparameters.

# COMMAND ----------

rfc.get_params()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train and Evaluate Model
# MAGIC
# MAGIC Now that we've set our hyperparameters to our preferred values, let's train and evaluate our model using accuracy.

# COMMAND ----------

from sklearn.metrics import accuracy_score

# Fit the model
rfc.fit(train_df.drop("steps_10000", axis=1), train_df["steps_10000"])

# Train accuracy
train_accuracy = accuracy_score(
  train_df["steps_10000"], 
  rfc.predict(train_df.drop("steps_10000", axis=1))
)

# Test accuracy
test_accuracy = accuracy_score(
  test_df["steps_10000"], 
  rfc.predict(test_df.drop("steps_10000", axis=1))
)
print("Train accuracy:", train_accuracy)
print("Test accuracy:", test_accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC Given our pretty small forest and really shallow trees, we're seeing that we're slightly underfitting our training set.

# COMMAND ----------

# MAGIC %md
# MAGIC ### New Hyperparameter Values
# MAGIC Let's change our hyperparameter values to see if we can get a better accuracy.

# COMMAND ----------

rfc.set_params(max_depth=5, n_estimators=10)

# COMMAND ----------

# Fit the model
rfc.fit(train_df.drop("steps_10000", axis=1), train_df["steps_10000"])

# Train accuracy
train_accuracy = accuracy_score(
  train_df["steps_10000"], 
  rfc.predict(train_df.drop("steps_10000", axis=1))
)

# Test accuracy
test_accuracy = accuracy_score(
  test_df["steps_10000"], 
  rfc.predict(test_df.drop("steps_10000", axis=1))
)
print("Train accuracy:", train_accuracy)
print("Test accuracy:", test_accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC That's a little better.

# COMMAND ----------

# MAGIC %md
# MAGIC ### One More Time
# MAGIC
# MAGIC Let's try one more time.

# COMMAND ----------

rfc.set_params(max_depth=8, n_estimators=100)

# Fit the model
rfc.fit(train_df.drop("steps_10000", axis=1), train_df["steps_10000"])

# Train accuracy
train_accuracy = accuracy_score(
  train_df["steps_10000"], 
  rfc.predict(train_df.drop("steps_10000", axis=1))
)

# Test accuracy
test_accuracy = accuracy_score(
  test_df["steps_10000"], 
  rfc.predict(test_df.drop("steps_10000", axis=1))
)
print("Train accuracy:", train_accuracy)
print("Test accuracy:", test_accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC And that's even better! Hopefully it's clear how changing the hyperparameter values can affect the training process and, as a result, the performance of the model.
# MAGIC
# MAGIC **Question:** How could we determine the optimal values for our hyperparameters?

# COMMAND ----------

# MAGIC %md
# MAGIC Through the rest of this lesson, we'll look at how to optimize these values.
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>