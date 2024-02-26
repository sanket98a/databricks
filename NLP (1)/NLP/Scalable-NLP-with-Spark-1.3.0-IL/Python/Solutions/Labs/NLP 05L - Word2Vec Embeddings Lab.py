# Databricks notebook source
# MAGIC
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Word2Vec Embeddings Lab
# MAGIC
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lab you:<br>
# MAGIC * Build a classification model using Word2Vec embeddings

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating Word2Vec Embeddings

# COMMAND ----------

# MAGIC %md
# MAGIC Create a column called `word2vecEmbedding` (using <a href="https://spark.apache.org/docs/latest/ml-features.html#word2vec" target="_blank"> SparkML's Word2Vec </a>) to use as features to train our next logistic regression model.
# MAGIC
# MAGIC Since this cell takes longer to run due to the large number of rows, we will first sample our DataFrame and train on that subset of our data using the `.sample()` function before applying the Word2Vec model to the entire `processedDF`.

# COMMAND ----------

from pyspark.ml.feature import Binarizer
from pyspark.sql.functions import col

processedDF = spark.read.parquet("/mnt/training/reviews/tfidf.parquet").withColumn(
    "Score", col("Score").cast("double")
)
binarizer = Binarizer(threshold=4.0, inputCol="Score", outputCol="ExcellentReview")
binarizedDF = binarizer.transform(processedDF)
shortenedDF = binarizedDF.sample(False, 0.3, 42)

# COMMAND ----------

from pyspark.ml.feature import Word2Vec

word2Vec = Word2Vec(
    vectorSize=20, minCount=2, inputCol="CleanTokens", outputCol="word2vecEmbedding"
)
model = word2Vec.fit(shortenedDF)

wvDF = model.transform(shortenedDF)
display(wvDF.select("Text", "word2vecEmbedding", "ExcellentReview"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Now we would like to train another logistic regression model except now we want it to predict the `ExcellentReview` column given the Spark Word2Vec embedding of the text.
# MAGIC
# MAGIC First split our dataset, `wvDF`, into a train and a test DataFrame. Don't forget to set a seed value and pick a reasonable train test split percentage.

# COMMAND ----------

# ANSWER
(trainDF, testDF) = wvDF.randomSplit([0.8, 0.2], seed=42)

# Cache split dfs
trainDF.cache().count()
testDF.cache().count()

# COMMAND ----------

# MAGIC %md
# MAGIC Fill in the below cell to create and fit the logistic regression model! Make sure that your model uses the `word2vecEmbedding` column to predict the label in the `ExcellentReview` column. Feel free to adjust the hyperparameters of the <a href="https://spark.apache.org/docs/latest/ml-classification-regression.html#binomial-logistic-regression" target="_blank">logistic regression</a> model to try to increase performance.

# COMMAND ----------

# ANSWER
from pyspark.ml.classification import LogisticRegression

# Create the model
lr_embeddings = LogisticRegression(
    featuresCol="word2vecEmbedding", labelCol="ExcellentReview"
)

# Fit the model
lrModel_embeddings = lr_embeddings.fit(trainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC After building and training our model, let's evaluate it! Fill in the `accuracy`, `falsePositiveRate`, and `val_acc` metrics of your trained model.

# COMMAND ----------

# ANSWER
# Evaluate model
trainingSummary = lrModel_embeddings.summary
accuracy = trainingSummary.accuracy
falsePositiveRate = trainingSummary.weightedFalsePositiveRate
val_acc = (
    lrModel_embeddings.transform(testDF).where("ExcellentReview = prediction").count()
    / testDF.count()
)

print(
    f"Training Accuracy: {accuracy}\nFPR: {falsePositiveRate}\nValidation Accuracy:{val_acc}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC Run the next cell to see what your new logistic regression model predicted for some specific text columns.

# COMMAND ----------

display(
    lrModel_embeddings.transform(testDF).select(
        ["Text", "ExcellentReview", "prediction"]
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC We would like to compare the 2 models! Fill out the following cell to join 2 dataframes so that we can see the `Text`, `ExcellentReview`, and `prediction` columns from both logistic regression models.

# COMMAND ----------

# load the predictions from the model that used TFIDF scores
tfidf_predictions = spark.read.parquet(
    "/mnt/training/reviews/tfidf_predictions.parquet"
)

# COMMAND ----------

# ANSWER
from pyspark.sql.functions import col

model1_df = tfidf_predictions.select(
    ["Text", "ExcellentReview", col("prediction").alias("prediction (tfidf)")]
)
model2_df = lrModel_embeddings.transform(testDF).select(
    ["Text", "ExcellentReview", col("prediction").alias("prediction (embedding)")]
)
combined_df = model1_df.join(model2_df, ["Text", "ExcellentReview"]).dropDuplicates()
display(combined_df)

# COMMAND ----------

# MAGIC %md
# MAGIC What do you notice about the predictions of both models compared to the actual score?
# MAGIC
# MAGIC Feel free to try training a different <a href="https://spark.apache.org/docs/latest/ml-classification-regression.html#classification"> classification model</a> on our data!
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>