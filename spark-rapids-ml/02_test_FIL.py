#!/usr/bin/env python



import os, warnings, sys, logging
import mlflow
import pandas as pd
import numpy as np
from datetime import date
# use the GPU-native implementation
from spark_rapids_ml.classification import RandomForestClassifier

from pyspark.sql import SparkSession

import pprint

# Force-clear any hanging Py4J connections
from py4j.java_gateway import java_import




USERNAME = os.environ["PROJECT_OWNER"]
DBNAME = "DEMO_"+USERNAME
CONNECTION_NAME = "go01-aw-dl"
STORAGE =  os.environ["DATA_STORAGE"] 
DATE = date.today()

RAPIDS_JAR = "/home/cdsw/rapids-4-spark_2.12-25.10.0.jar"


LOCAL_PACKAGES = "/home/cdsw/.local/lib/python3.10/site-packages"
# This is where the specific CUDA 12 NVRTC library lives
NVRTC_LIB_PATH = f"{LOCAL_PACKAGES}/nvidia/cuda_nvrtc/lib"
WRITABLE_CACHE_DIR = "/tmp/cupy_cache"





spark = SparkSession.builder \
    .appName("Spark-Rapids-32GB-Final") \
    .config("spark.jars", RAPIDS_JAR) \
    .config("spark.plugins", "com.nvidia.spark.SQLPlugin") \
    .config("spark.executor.resource.gpu.vendor", "nvidia.com") \
    .config("spark.executor.resource.gpu.discoveryScript", "/home/cdsw/spark-rapids-ml/getGpusResources.sh") \
    .config("spark.executorEnv.LD_LIBRARY_PATH", f"{NVRTC_LIB_PATH}:{os.environ.get('LD_LIBRARY_PATH', '')}") \
    .config("spark.executorEnv.PYTHONPATH", LOCAL_PACKAGES) \
    .config("spark.executorEnv.CUPY_CACHE_DIR", WRITABLE_CACHE_DIR) \
    .config("spark.driverEnv.CUPY_CACHE_DIR", WRITABLE_CACHE_DIR) \
    .config("spark.driver.memory", "12g") \
    .config("spark.driver.extraJavaOptions", f"-Djava.library.path={NVRTC_LIB_PATH}") \
    .config("spark.driver.maxResultSize", "4g") \
    .config("spark.dynamicAllocation.enabled", "false") \
    .config("spark.executor.cores", 2) \
    .config("spark.executor.instances", 1) \
    .config("spark.executor.heartbeatInterval", "60s") \
    .config("spark.executor.memory", "10g") \
    .config("spark.executor.resource.gpu.amount", 1) \
    .config("spark.executor.memoryOverhead", "10g") \
    .config("spark.sql.autoBroadcastJoinThreshold", -1) \
    .config("spark.sql.broadcastTimeout", "1200") \
    .config("spark.sql.cache.serializer", "com.nvidia.spark.ParquetCachedBatchSerializer") \
    .config('spark.sql.shuffle.partitions', '200') \
    .config("spark.network.timeout", "800s") \
    .config("spark.rapids.sql.enabled", "true") \
    .config("spark.rapids.shims-provider-override", "com.nvidia.spark.rapids.shims.spark351.SparkShimServiceProvider") \
    .config("spark.rapids.memory.pinnedPool.size", "4g") \
    .config("spark.task.resource.gpu.amount", 0.5) \
    .config("spark.kerberos.access.hadoopFileSystems", "s3a://go01-demo/user/jprosser/spark-rapids-ml/") \
    .config("spark.shuffle.service.enabled", "false") \
    .config('spark.shuffle.file.buffer', '64k') \
    .config('spark.shuffle.spill.compress', 'true') \
    .config("spark.hadoop.fs.defaultFS", "s3a://go01-demo/") \
    .getOrCreate()



spark.sparkContext.setLogLevel("WARN")
# View the underlying Java Spark Context
pprint.pprint(f"Java Context Object: {spark.sparkContext._jsc}")

# View the Spark Master (in CML, this usually points to the local container or YARN)
pprint.pprint(f"Master: {spark.sparkContext.master}")

# View the User running the session
pprint.pprint(f"Spark User: {spark.sparkContext.sparkUser()}")



# Enable CollectLimit so that large datasets are collected on the GPU.
# Not worth it for small datasets
spark.conf.set("spark.rapids.sql.exec.CollectLimitExec", "true")

# Enabled to let the GPU to handle the random sampling of rows for large datasets
spark.conf.set("spark.rapids.sql.exec.SampleExec", "true")

# Enabled to let allow more time for large broadcast joins
spark.conf.set("spark.sql.broadcastTimeout", "1200") # Increase to 20 mins
from pyspark.sql import functions as F

#spark.conf.set("spark.rapids.sql.explain", "ALL")
spark.conf.set("spark.rapids.sql.explain", "NOT_ON_GPU") # Only log when/why the GPU was not selected
spark.conf.set("spark.rapids.sql.variable.float.allow", "true") # Allow float math

# Allow the GPU to cast instead of pushing back to CPU just for cast
spark.conf.set("spark.rapids.sql.castFloatToDouble.enabled", "true") 
spark.conf.set("spark.rapids.sql.format.parquet.enabled", "true")

# Turning off Adaptive Query Execution (AQE) makes the entire SQL plan use the GPU
spark.conf.set("spark.sql.adaptive.enabled", "false")


spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 10000)


# Test if the JVM can actually talk to the CUDA driver
cuda_manager = spark._jvm.ai.rapids.cudf.Cuda
print(f"CUDA Driver Version: {cuda_manager.getDriverVersion()}")
print(f"Device Count: {cuda_manager.getDeviceCount()}")
print(f"Dynamic Allocation: {spark.conf.get('spark.dynamicAllocation.enabled')}")
print(f"Executor Instances: {spark.conf.get('spark.executor.instances')}")
print(f"Dynamic Allocation Enabled: {spark.conf.get('spark.dynamicAllocation.enabled')}")


# Test acess to the SQLPlugin
sql_plugin = spark._jvm.com.nvidia.spark.SQLPlugin()

driver_comp = sql_plugin.driverPlugin()


log_manager = spark._jvm.org.apache.log4j.LogManager
level_debug = spark._jvm.org.apache.log4j.Level.DEBUG

logger = driver_comp.log()
log_manager.getLogger("com.nvidia.spark.rapids").setLevel(level_debug)

print(f"Debug enabled for RAPIDS: {driver_comp.isTraceEnabled() or True}")


df = spark.read.table("DataLakeTable")
print(f"Columns: {len(df.columns)}")
print(f"Schema: {df.schema}")
# Look for 'Gpu' operators in the output
df.limit(5).explain(mode="formatted")

# Transform data into a single vector column 
feature_cols = ["age", "credit_card_balance", "bank_account_balance", "mortgage_balance", "sec_bank_account_balance", "savings_account_balance",
                    "sec_savings_account_balance", "total_est_nworth", "primary_loan_balance", "secondary_loan_balance", "uni_loan_balance",
                    "longitude", "latitude", "transaction_amount"]

# Avoid VectorAssembler as it creates VectorUDT data types that are not GPU Friendly
#assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
#df_assembled = assembler.transform(df)


# Split data into training and test sets
#(training_data, test_data) = df_assembled.randomSplit([0.8, 0.2], seed=1234)

(training_data, test_data) = df.randomSplit([0.8, 0.2], seed=1234)

#from sklearn.model_selection import train_test_split
#X_train, X_validation, y_train, y_validation = train_test_split(
#    X, y, train_size=train_size)

# Use spark_rapids_ml.classification.RandomForestClassifier

# Import from spark_rapids_ml to use the GPU-native implementation
from spark_rapids_ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Define the RAPIDS-native classifier
# As noted above, by using 'featuresCols' (list of strings), we avoid VectorAssembler 
# 
rf_classifier = RandomForestClassifier(
    labelCol="fraud_trx", 
    featuresCols=feature_cols, 
    numTrees=20
)


# Train the model
# This runs the training logic in C++ on the GPU via cuML
print("Training Spark RAPIDS ML model...")
rf_model = rf_classifier.fit(training_data)
print("Model training complete.")
print(type(rf_model))

# Predict and optimize the output
# We drop 'probability' and 'rawPrediction' because they are VectorUDT types
# that Spark SQL would otherwise force back to the CPU for formatting.
predictions = rf_model.transform(test_data).drop("probability", "rawPrediction")
rf_model.setFeaturesCols(feature_cols)
# Show results (This will be fully accelerated)
predictions.select("prediction", "fraud_trx").show(5)
# Verify GPU Plan
# You should see 'GpuProject' and 'GpuFilter' nodes without the VectorUDT warning
predictions.explain(mode="formatted")


import spark_rapids_ml.metrics.MulticlassMetrics as mm
#print(f"Available in metrics: {help(mm)}")


accuracy = predictions.filter("prediction = fraud_trx").count() / predictions.count()

print(f"GPU-Accelerated Accuracy: {accuracy:.4f}")


#from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from spark_rapids_ml.metrics.MulticlassMetrics import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(
    labelCol="fraud_trx", 
    predictionCol="prediction", 
    metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy: {accuracy:.4f}")



from spark_rapids_ml.classification import RandomForestClassifier as cuRFC
from spark_rapids_ml.classification import RandomForestClassificationModel as cuRFCM
#print(f"Available in metrics: {help(cuRFCM)}")



treelite_model_checkpoint = rf_model._treelite_model

treelite_model_data = rf_model._treelite_model
print(type(treelite_model_data))
print("len:", len(treelite_model_data))
print("first20:", treelite_model_data[:20])


##pprint.pprint(treelite_model_data)
#    
#with open("fraud_rf_model.tl", "wb") as f:
#    f.write(treelite_model_data)

import treelite
import cuml
import spark_rapids_ml
print(f"Treelite: {treelite.__version__}")
print(f"cuML:     {cuml.__version__}")
print(f"spark_rapids_ml:     {spark_rapids_ml.__version__}")


# cuML v25.10.00 requires Treelite v4.4.1 but the serialized Treelite model it generates
# appears to be  v3.9
#main_model = treelite.Model.deserialize("fraud_rf_model.tl")



import base64
import pickle
import treelite


# 2. Decode from Base64 to Pickle-bytes
pickle_bytes = base64.b64decode(treelite_model_checkpoint)

# 3. Unpickle to get the actual Treelite binary bytes
# Note: This returns the raw bytes that Treelite actually understands
raw_treelite_bytes = pickle.loads(pickle_bytes)

# 4. Now deserialize using Treelite
treelite_model = treelite.Model.deserialize_bytes(raw_treelite_bytes)



from cuml.fil import ForestInference

fm = ForestInference.load_from_treelite_model(treelite_model)

print(f"Success! Model loaded with {treelite_model.num_tree} trees.")

test_data_vals=np.array(test_data.collect())

# Your model is now ready for GPU inference
predictions = fm.predict(test_data_vals)


# Done

import treelite
import onnxmltools
from skl2onnx import update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost

# Note: The exact import path for Treelite converter in skl2onnx 
# may vary, but here is the most stable approach:

# 1. Load your model as we did before
#treelite_model = treelite.Model.deserialize_bytes(raw_treelite_bytes)

# 2. Define input shape
initial_type = [('input', FloatTensorType([None, treelite_model.num_feature]))]

# 3. Use the general conversion utility
# In most recent versions, onnxmltools uses the XGBoost operator logic 
# to handle Treelite structures because they share the same 'TreeEnsemble' math.
try:
    from onnxmltools.convert import convert_treelite
    onnx_model = convert_treelite(treelite_model, initial_types=initial_type)
except AttributeError:
    # If the attribute is still missing, we use the 'XGBoost' path 
    # which is the underlying architecture of your Spark Rapids model
    from onnxmltools.convert import convert_xgboost
    onnx_model = convert_xgboost(model, initial_types=initial_type)

# 4. Save
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
