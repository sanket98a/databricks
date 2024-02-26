# Databricks notebook source
# MAGIC
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Sentiment Analysis Lab
# MAGIC
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lab you:<br>
# MAGIC * Analyze the sentiment of each review using `vaderSentiment` and `textblob`

# COMMAND ----------

# MAGIC %pip install vaderSentiment==3.2.1

# COMMAND ----------

# MAGIC %pip install textblob==0.15.3

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

processedDF = spark.read.parquet("/mnt/training/reviews/tfidf.parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC Another extremely useful functionality that NLTK has (in its [vaderSentiment](https://www.nltk.org/api/nltk.sentiment.html)  package)
# MAGIC is sentiment analysis. Sentiment analysis is the process of using NLP to systematically identify, extract, quantify, and study affective states and subjective information. Often times this means trying to identify whether a piece of text has an overall positive, neutral, or negative connotation to it.
# MAGIC
# MAGIC `vaderSentiment.SentimentIntensityAnalyzer` takes in raw unprocessed text and returns a `float` between -1 and 1 where scores closer to 1 indicate positive sentiment, scores closer to -1 indicate negative sentiment, and scores close to 0 indicate neutral sentiment.
# MAGIC
# MAGIC Below, we have defined a helper function `get_nltk_sentiment_score` which will take in one string of text and return a float representing the overall sentiment of that string. Use this function to create a UDF that will add a column called `sentimentNLTKScore` to `processedDF`.

# COMMAND ----------

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def get_nltk_sentiment_score(text):
    return analyzer.polarity_scores(text)["compound"]


# Example usage
get_nltk_sentiment_score("The function will return the sentiment of this sentence!")

# COMMAND ----------

# TODO
from pyspark.sql.types import *
from pyspark.sql.functions import col, udf

@udf("double")
def sentiment_nltk_udf(text):
    return analyzer.polarity_scores(text)["compound"]

sentimentDF = processedDF.withColumn("sentimentNLTKScore",sentiment_nltk_udf(col("Text")))

display(sentimentDF.select("Text", "Score", "sentimentNLTKScore"))

# COMMAND ----------

# MAGIC %md
# MAGIC Let's see how the NLTK sentiment score of a review is related to the `Score` that the reviewer gave!
# MAGIC
# MAGIC Instead of looking at the dataframe in a table, we will take a look at the scatterplot between `sentimentNLTKScore` and `Score` of each review.
# MAGIC 1. Run the following display command on the `Score` and `sentimentNLTKScore` columns of the `sentimentDF`
# MAGIC 2. Click on the middle (bar graph) icon on the bottom left of the cell output
# MAGIC 3. Click on the plot options to open it
# MAGIC 4. Change "Display type" to "Scatter Plot", check the box next to "Show LOESS"
# MAGIC 5. Drag and drop `sentimentNLTKScore` then `Score` into the "Values" box
# MAGIC 6. Click "Apply" on the bottom right corner

# COMMAND ----------

display(sentimentDF.select("Score", "sentimentNLTKScore"))

# COMMAND ----------

# MAGIC %md
# MAGIC From the LOESS curve we can see that the sentiment of the review text is directly correlated with the score that a product is given. This is good news! It means that there is information within the `Text` column of the dataframe that we can use to predict the corresponding `Score` the product is given.

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's try to compare the NLTK sentiment analyzer to that of <a href="https://textblob.readthedocs.io/en/dev/#" target="_blank">TextBlob</a>.
# MAGIC
# MAGIC Again we have defined for you a helper function, `get_textblob_sentiment_score()`, which takes in 1 string of text and returns a tuple of floats representing the overall sentiment of that string. The textblob library defines polarity as a float within the range [-1.0, 1.0] and subjectivity as a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective.
# MAGIC
# MAGIC Use this function to create a UDF that will add a column called `sentimentTextBlobScore` to the `sentimentDF` you created above.

# COMMAND ----------

from textblob import TextBlob


def get_textblob_sentiment_score(text):
    sentiment = TextBlob(text).sentiment
    return (sentiment.polarity, sentiment.subjectivity)


# Example usage
get_textblob_sentiment_score("The function will return the sentiment of this sentence!")

# COMMAND ----------

# TODO
from pyspark.sql.types import *
from pyspark.sql.functions import col, udf

text_blob_data_type = StructType(
    [
        StructField("polarity", DoubleType()),
        StructField("subjectivity", DoubleType()),
    ]
)

@udf(text_blob_data_type)
def sentiment_textblob_udf(text):
    sentiment = TextBlob(text).sentiment
    return sentiment.polarity, sentiment.subjectivity

sentimentDF = sentimentDF.withColumn("sentimentTextBlobScore",sentiment_textblob_udf(col("Text")))
display(sentimentDF.select("Text", "Score", "sentimentNLTKScore", "sentimentTextBlobScore"))

# COMMAND ----------

# MAGIC %md
# MAGIC Compare and contrast the sentiment scores given by these 2 libraries.
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>