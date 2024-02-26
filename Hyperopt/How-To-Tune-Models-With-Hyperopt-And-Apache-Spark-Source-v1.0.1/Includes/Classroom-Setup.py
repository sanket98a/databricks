# Databricks notebook source
import pandas as pd
import re
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning) 

None # Suppress Output

# COMMAND ----------

# Data transfer utility functions
# The following utility functions transfer course data from their central location to the learners working directory.

def path_exists(path):
  try:
    return len(dbutils.fs.ls(path)) >= 0
  except Exception:
    return False

def install_datasets(working_dir, course_name, version, min_time, max_time, reinstall=False):
  print(f"Your working directory is\n{working_dir}\n")

  # You can swap out the source_path with an alternate version during development
  # source_path = f"dbfs:/mnt/work-xxx/{course_name}"
  source_path = f"wasbs://courseware@dbacademy.blob.core.windows.net/{course_name}/{version}"
  print(f"The source for this dataset is\n{source_path}/\n")
  
  # Change the final directory to another name if you like, e.g. from "datasets" to "raw"
  target_path = f"{working_dir}/datasets"
  existing = path_exists(target_path)

  if not reinstall and existing:
    print(f"Skipping install of existing dataset to\n{target_path}")
    return 

  # Remove old versions of the previously installed datasets
  if existing:
    print(f"Removing previously installed datasets from\n{target_path}")
    dbutils.fs.rm(target_path, True)
  
  print(f"""Installing the datasets to {target_path}""")
  
  print(f"""\nNOTE: The datasets that we are installing are located in Washington, USA - depending on the
      region that your workspace is in, this operation can take as little as {min_time} and 
      upwards to {max_time}, but this is a one-time operation.""")

  dbutils.fs.cp(source_path, target_path, True)
  print(f"""\nThe install of the datasets completed successfully.""")  
  
None # Suppress Output

# COMMAND ----------

# User/Course set-up variables

# The following code is 100% scafolding only and can be found in Classroom-Setup.
# There is no need to copy any of this code.
course_name = "how-to-tune-models-with-hyperopt-and-apache-spark"
clean_course_name = re.sub("[^a-zA-Z0-9]", "_", course_name).lower() # sanitize the course name for use in directory and database names

lesson_name = dbutils.entry_point.getDbutils().notebook().getContext().notebookPath().getOrElse(None).split("/")[-1]
clean_lesson_name = re.sub("[^a-zA-Z0-9]", "_", lesson_name).lower() # sanitize the lesson name for use in directory and 

username = spark.sql("SELECT current_user()").collect()[0][0]
clean_username = re.sub("[^a-zA-Z0-9]", "_", username)

database = f"dbacademy_{clean_username}_httmwhaas"
spark.sql(f"CREATE DATABASE IF NOT EXISTS {database}")
spark.sql(f"USE {database}")

root_working_dir = f"dbfs:/user/{username}/dbacademy/{clean_course_name}"
working_dir = f"{root_working_dir}/{clean_lesson_name}"

def classroom_cleanup():
    spark.sql(f"DROP DATABASE IF EXISTS {database}")
    dbutils.fs.rm(working_dir, True)
    
None # Suppress Output

# COMMAND ----------

# Data transfer

# Note the course name passed here identifies the directory within dbacademy/courseware
# to install datsaets from and need not match the course_name from above - but it should ;-)
install_datasets(root_working_dir, course_name, "v01", "3 seconds", "3 minutes")

None # Suppress Output

# COMMAND ----------

# Validate data transfer

# If you want to go the extra mile, you can validate that your datasets were installed correctly
def validate_file_count(path, expected_count):
    actual_files = dbutils.fs.ls(path)
    assert len(actual_files) == expected_count, f"Expected {expected_count} files, found {len(actual_files)}"


validate_file_count(f"{root_working_dir}/datasets", 2)

# COMMAND ----------

# %fs ls "dbfs:/user/sanket.bodake@affine.ai/dbacademy/how_to_tune_models_with_hyperopt_and_apache_spark/datasets/winequality-white.csv"

# COMMAND ----------

# df=spark.read.csv("dbfs:/user/sanket.bodake@affine.ai/dbacademy/how_to_tune_models_with_hyperopt_and_apache_spark/datasets/winequality-white.csv"
# ,sep=';',header=True)

# COMMAND ----------

# df.toPandas()

# COMMAND ----------

# Data loading utility function
# Data load and prep utility function that the learner can use to easily load and prepare data for analysis.

def get_wine_data():
    """
    Prepare wine quality dataset for Hyperopt course.
  
    Automates the process of preparing the well known wine quality dataset 
    (https://archive.ics.uci.edu/ml/datasets/wine+quality) provided by the 
    UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/index.php). 
    We prepare this data by:

       1. Reading in the data
       2. Creating an indicator column for red vs. white wine
       3. Combing the red and white wine datasets
       4. Cleaning up column names
       5. Converting wine quality column to a binary response variable
          - 1: quality >= 7
          - 0: quality < 7
    """
    # 1. read in data
    data_path = f"{root_working_dir.replace('dbfs:', '')}/datasets"
    print(data_path)
    white_wine = spark.read.csv(f"{data_path}/winequality-white.csv", sep=";",header=True,inferSchema=True)
    red_wine = spark.read.csv(f"{data_path}/winequality-red.csv", sep=";",header=True,inferSchema=True)
    white_wine=white_wine.toPandas()
    red_wine=red_wine.toPandas()
    
    # 2. create indicator column for red vs. white wine
    red_wine['is_red'] = 1
    white_wine['is_red'] = 0
    
    # 3. combine the red and white wine data sets
    data = pd.concat([red_wine, white_wine], axis=0)
    
    # 4. remove spaces from column names
    data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)
    # 5. convert "quality" column to 0 vs. 1 to make this a classification problem
    data["quality"] = (data["quality"].astype(int)>= 7)
    
    return data

# COMMAND ----------

print("Setup Complete. You can now use `get_wine_data()` to install the required dataset.")


# COMMAND ----------

# get_wine_data()

# COMMAND ----------

data = get_wine_data()

# COMMAND ----------

