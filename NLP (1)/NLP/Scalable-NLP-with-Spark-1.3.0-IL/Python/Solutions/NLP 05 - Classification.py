# Databricks notebook source
# MAGIC
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Classification
# MAGIC
# MAGIC This lesson you will learn different ways of using the text column of the Food Reviews Dataset to predict the score of the review.
# MAGIC
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC * Build a classification model using TFIDF scores

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC First run the following cell to load the dataframe with our previously calculated TFIDF scores.

# COMMAND ----------

processedDF = spark.read.parquet("/mnt/training/reviews/tfidf.parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The first thing we should do is take a look at the `Score` column to get a feel for what our model will try to predict.
# MAGIC
# MAGIC Let's view the distribution of scores in our dataset. Use the Databricks visualization to display a bar graph of how many of each score is present in our data.

# COMMAND ----------

display(processedDF.groupBy("Score").count().orderBy("count", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC We can see that the data is heavily skewed. More than half of the reviews scores are a 5.
# MAGIC
# MAGIC To simplify the problem a little, we would like to predict whether the text of a review corresponds to an 'excellent' or a 'not excellent' review rating. We are going to create a new column called `ExcellentReview` and put a `1` for an excellent review and a `0` for a not excellent review. For our purposes, an excellent review is one whose `score` is equal to `5`. This grouping is not nearly as skewed since both classes now have similar number of reviews.

# COMMAND ----------

from pyspark.sql.types import IntegerType
from pyspark.sql.functions import *

excellentReviewUDF = udf(lambda score: 1 if int(score) == 5 else 0, IntegerType())

classificationDF = processedDF.withColumn(
    "ExcellentReview", excellentReviewUDF(col("Score"))
)
display(classificationDF.groupBy("ExcellentReview").count().sort(col("count").desc()))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Training with TFIDF Scores
# MAGIC
# MAGIC Now we will split our dataframe into training and testing dataframes and use those to create and evaluate a logistic regression model. We are using the <a href="https://spark.apache.org/docs/latest/ml-classification-regression.html#binomial-logistic-regression" target="_blank">logistic regression</a> model from SparkML and will first train it to predict the `ExcellentReview` value given the TFIDF score (`TFIDFScoreNorm` value) of a review.
# MAGIC
# MAGIC We'll split our data into training and test datasets. This is necessary because we want to use the unseen data to verify that our model generalizes well on unseen data. We will use 80% for training and the remaining 20% for testing and set a seed so that the results are reproducible (i.e. if you re-run this notebook, you'll get the same results each times).

# COMMAND ----------

# Train test split
trainDF, testDF = classificationDF.randomSplit([0.8, 0.2], seed=42)
print(trainDF.cache().count())
print(testDF.cache().count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Logistic Regression
# MAGIC
# MAGIC A logistic regression model can be used to predict binary outputs, in our case `0` for not a excellent review and `1` for a excellent review. It uses the logistic function:
# MAGIC $$\sigma(t) = \frac{1}{1+e^{-t}}$$
# MAGIC
# MAGIC ![](https://files.training.databricks.com/images/logistic.png)
# MAGIC
# MAGIC As seen in the graph, the output of the logistic function is always between 0 and 1, making its outputs interpretable as probabilities. An output of 0.1 means that the model thinks there is a 10% chance of being a label `1` so the model will assign the input point a label of `0`. Similarly, an output of 0.9 means that there is a 90% chance of being a label `1` so we will assign this input a label of `1`.
# MAGIC
# MAGIC Under the hood, the logistic regression model first fits a linear regression model to the data then applies the logistic function to the linear prediction to determine the output label of any data point.

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(featuresCol="TFIDFScoreNorm", labelCol="ExcellentReview")

# Fit the model
lrModel = lr.fit(trainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation Metrics
# MAGIC We can evaluate our binary classification model using several metrics. Some metrics include
# MAGIC
# MAGIC * `Training Accuracy`: The percentage of reviews where the model correctly predicted if the review would be Excellent or not on the `train` set.
# MAGIC * `False Positive Rate`: The total number of false positives divided by the number of true negatives plus false positives. False positives are reviews incorrectly predicted as excellent. The true negative rate is the number of reviews correctly predicted as not excellent. $$\frac{FP}{FP + TN} $$
# MAGIC * `Validation Accuracy`: The percentage of reviews where the model correctly predicted if the review would be Excellent or not on the `validation` set.

# COMMAND ----------

# Evaluate model
trainingSummary = lrModel.summary
accuracy = trainingSummary.accuracy
falsePositiveRate = trainingSummary.weightedFalsePositiveRate
val_acc = (
    lrModel.transform(testDF).where("ExcellentReview = prediction").count()
    / testDF.count()
)

print(
    f"Training Accuracy: {accuracy}\nFPR: {falsePositiveRate}\nValidation Accuracy:{val_acc}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC Now take a look at what our model predicted for some specific text columns from the data we saved for testing purposes, `testDF_tfidf`.

# COMMAND ----------

tfidf_predictions_df = lrModel.transform(testDF).select(
    ["Text", "ExcellentReview", "prediction"]
)
display(tfidf_predictions_df)

# COMMAND ----------

# MAGIC %md
# MAGIC We can see that it did pretty well! This means that the TFIDF score of our text did a good job retaining information about how "excellent" a review is.
# MAGIC
# MAGIC Save the predictions of this classification model.

# COMMAND ----------

path = f"{workingDir}/tfidf_predictions.parquet"
tfidf_predictions_df.repartition(8).write.mode("OVERWRITE").parquet(path)


# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>