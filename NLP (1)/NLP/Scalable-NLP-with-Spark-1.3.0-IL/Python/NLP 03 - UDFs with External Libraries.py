# Databricks notebook source
# MAGIC
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Using External Libraries with UDFs
# MAGIC
# MAGIC This lesson introduces more advanced text processing steps by applying single-node libraries in parallel for feature preprocessing.
# MAGIC
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC * Learn about single node NLP libraries
# MAGIC * Create user defined functions to parallelize library calls
# MAGIC * Lemmatize and stem your tokens

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Single-Machine Natural Language Processing Libraries
# MAGIC
# MAGIC Just because a library is built for a single node doesn't mean you can't apply it to your code in parallel! In general, anything that is a rule-based transformation can be applied in parallel.
# MAGIC
# MAGIC Below is a list of popular single node NLP libraries:
# MAGIC * <a href="https://www.nltk.org/" target="_blank">NLTK</a>
# MAGIC * <a href="https://spacy.io/" target="_blank">spaCy</a>
# MAGIC * <a href="https://textblob.readthedocs.io/en/dev/#" target="_blank">TextBlob</a>
# MAGIC * <a href="https://stanfordnlp.github.io/CoreNLP/" target="_blank">Stanford CoreNLP</a>
# MAGIC * <a href="https://opennlp.apache.org/" target="_blank">OpenNLP</a>
# MAGIC * <a href="https://radimrehurek.com/gensim/" target="_blank">Gensim</a>
# MAGIC * <a href="https://allennlp.org/" target="_blank">AllenNLP</a>
# MAGIC
# MAGIC We will be using `NLTK`, `vaderSentiment`, and `TextBlob` in this lesson.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## User Defined Function (UDF)
# MAGIC
# MAGIC While the distributed libraries we looked at in the last lesson are optimized for speed + distributed computation, sometimes they don't have all the functionalities that we need. In those cases, we will have to rely on more developed single-node libraries. We can write a **user defined function** to apply these libraries to our DataFrame in parallel.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stemming and Lemmatizing with NLTK
# MAGIC
# MAGIC We discussed earlier how to make our strings more comparable by tokenizing and lowercasing. But so far, strings like "run", "running", and "ran" are still treated as completely unrelated words if we were to compare them; however, their only difference is purely a result of grammatical structure. To resolve this, we are going to use a process called **stemming** which is going to 'chop' off the ends of words to get it to its base form. Since stemming only removes letters from our string, often times it results in strings that aren't real words.
# MAGIC
# MAGIC Another more involved process that attempts to change strings into more comparable forms is called **lemmatizing**. It tries to find the dictionary form - also called the lemma - of a word. This means the results of lemmatizing are real words that we recognize.

# COMMAND ----------

# MAGIC %pip install nltk

# COMMAND ----------

# Stemming vs Lemmatizing
import nltk

nltk.download("wordnet")
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["running", "ponies", "pony", "dogs", "people", "geese"]
word_forms = [(word, stemmer.stem(word), lemmatizer.lemmatize(word)) for word in words]

for orig, stemmed, lemmatized in word_forms:
    print(f"Original: {orig}  \tStemmed: {stemmed} \tLemmatized:{lemmatized}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Now that we understand the concept of stemming and lemmatizing and how to call a function as a UDF, we're going combine these skills to stem and lemmatize our reviews dataset's `CleanTokens` using the NLTK functions `PorterStemmer` and `WordNetLemmatizer`.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Stemming UDF

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##### For pandas dataframe

# COMMAND ----------

## Pandas DF
import pandas as pd

processedDF = spark.read.parquet("/mnt/training/reviews/tfidf.parquet")
pdDF = processedDF.limit(30).toPandas()


def stem_udf(tokens: pd.Series) -> pd.Series:
    ps = PorterStemmer()
    return [ps.stem(token) for token in tokens]


pdDF["StemTokens"] = pdDF["CleanTokens"].apply(stem_udf)

pdDF.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Notice the use of Python type hints above.
# MAGIC
# MAGIC Python type hints were officially introduced in PEP 484 with Python 3.5. Type hinting is an official way to statically indicate the type of a value in Python.
# MAGIC
# MAGIC Python type hints bring two significant benefits to the PySpark and Pandas UDF context:
# MAGIC * It gives a clear definition of what the function is supposed to do, making it easier for users to understand the code. It can avoid the need to document such subtle cases with a bunch of test cases and/or for users to test and figure out by themselves.
# MAGIC * It can make it easier to perform static analysis. IDEs such as PyCharm and Visual Studio Code can leverage type annotations to provide code completion, show errors, and support better go-to-definition functionality.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### For Spark DataFrame

# COMMAND ----------

# MAGIC %md
# MAGIC In Spark 2.4, this is an example of how we can write UDFs, without using type hints.

# COMMAND ----------

from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.functions import col

processedDF = spark.read.parquet("/mnt/training/reviews/tfidf.parquet")

ps = PorterStemmer()
stem_udf = udf(
    lambda tokens: [ps.stem(token) for token in tokens], ArrayType(StringType())
)

# add StemTokens column
stemmedDF = processedDF.withColumn("StemTokens", stem_udf(col("CleanTokens")))
display(stemmedDF.select("StemTokens", "Tokens"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC UDFs provide more functionality, but it is best to use a built-in function wherever possible.
# MAGIC
# MAGIC **UDF Drawbacks:**
# MAGIC * UDFs cannot be optimized by the Catalyst Optimizer
# MAGIC * The function **has to be serialized** and sent out to the executors
# MAGIC * In the case of Python, there is even more overhead - we have to **spin up a Python interpreter** on every Executor to run the UDF (e.g. Python UDFs much slower than Scala UDFs)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Vectorized UDF
# MAGIC
# MAGIC As of Spark 2.3, there are Vectorized UDFs available in Python to help speed up the computation.
# MAGIC
# MAGIC * [Blog post](https://databricks.com/blog/2017/10/30/introducing-vectorized-udfs-for-pyspark.html)
# MAGIC * [Documentation](https://spark.apache.org/docs/latest/sql-programming-guide.html#pyspark-usage-guide-for-pandas-with-apache-arrow)
# MAGIC
# MAGIC <img src="https://databricks.com/wp-content/uploads/2017/10/image1-4.png" alt="Benchmark" width ="500" height="1500">
# MAGIC
# MAGIC Vectorized UDFs utilize Apache Arrow to speed up computation. Let's see how that helps improve our processing time.
# MAGIC
# MAGIC The user-defined functions are executed by:
# MAGIC * [Apache Arrow](https://arrow.apache.org/), is an in-memory columnar data format that is used in Spark to efficiently transfer data between JVM and Python processes with near-zero (de)serialization cost. See more [here](https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html).
# MAGIC * pandas inside the function, to work with pandas instances and APIs.

# COMMAND ----------

# MAGIC %md
# MAGIC On the left, this is how a function is applied to a series, e.g. `Series.apply(..., axis='index')`. On the right, `pandas_udf` is applying the function to a batch of `value`s. For instance, in our case, each `value` here is an array of strings or tokens.
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/301/pandas_udf_2.png" height="600" width ="400">
# MAGIC
# MAGIC `pandas_udf` works similarly with pandas's `Dataframe.apply(..., axis='index')`
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/301/pandas_udf_1.png" height="600" width ="400">

# COMMAND ----------

import numpy as np
from pyspark.sql.functions import pandas_udf

# In order to correctly use pandas_udf, the return type should be specified
@pandas_udf(ArrayType(StringType()))
def stem_udf(tokens_batch: pd.Series) -> pd.Series:
    # `tokens_batch` looks like `pd.Series([["token"], ["token", "token"]])`
    ps = PorterStemmer()
    transformed = []
    for tokens in tokens_batch:
        transformed.append(np.array([ps.stem(token) for token in tokens]))
    # the output also has to be in the format of `pd.Series([["token"], ["token", "token"]])`
    return pd.Series(transformed)


# COMMAND ----------

# concise version
@pandas_udf(ArrayType(StringType()))
def stem_udf(tokens_batch: pd.Series) -> pd.Series:
    ps = PorterStemmer()
    return pd.Series(
        np.array([ps.stem(token) for token in tokens]) for tokens in tokens_batch
    )


# add StemTokens column
stemmedDF = processedDF.withColumn("StemTokens", stem_udf(col("CleanTokens")))
display(stemmedDF.select("StemTokens", "CleanTokens"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Lemmatizing UDF

# COMMAND ----------

# Method 1 without using pandas_udf

lemmatizer = WordNetLemmatizer()

# create UDF
@udf(ArrayType(StringType()))
def lemma_udf(tokens):
    nltk.download("wordnet")
    return [lemmatizer.lemmatize(token) for token in tokens]


# add LemmaTokens column
lemmaDF = processedDF.withColumn("LemmaTokens", lemma_udf(col("CleanTokens")))
display(lemmaDF.select("Tokens", "LemmaTokens"))

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we've gotten regular `udf` to work, let's use `pandas_udf` to speed up the computation!

# COMMAND ----------

# Method 2 using pandas_udf

lemmatizer = WordNetLemmatizer()


@pandas_udf(ArrayType(StringType()))
def lemma_udf(tokens_batch: pd.Series) -> pd.Series:
    nltk.download("wordnet")
    return pd.Series(
        np.array([lemmatizer.lemmatize(token) for token in tokens])
        for tokens in tokens_batch
    )


# add LemmaTokens column
lemmaDF = processedDF.withColumn("LemmaTokens", lemma_udf(col("CleanTokens")))
display(lemmaDF.select("LemmaTokens", "Tokens"))


# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>