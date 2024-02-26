# Databricks notebook source

spark.conf.set("com.databricks.training.module-name", "nlp")

# filter out warnings from python
# issue: https://github.com/RaRe-Technologies/smart_open/issues/319
import warnings
warnings.filterwarnings("ignore")

displayHTML("Preparing the learning environment...")

# COMMAND ----------

# MAGIC %run "./Class-Utility-Methods"

# COMMAND ----------

# MAGIC %run "./Dataset-Mounts"

# COMMAND ----------

courseType = "il"
username = getUsername()
userhome = getUserhome()
workingDir = getWorkingDir(courseType).replace("_pil", "")

# COMMAND ----------

courseAdvertisements = dict()
courseAdvertisements["username"] = (
    "v",
    username,
    "No additional information was provided.",
)
courseAdvertisements["userhome"] = (
    "v",
    userhome,
    "No additional information was provided.",
)
courseAdvertisements["workingDir"] = (
    "v",
    workingDir,
    "No additional information was provided.",
)
allDone(courseAdvertisements)
