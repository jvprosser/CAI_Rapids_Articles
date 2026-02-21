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
    .config("spark.executor.cores", 3) \
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
    .config("spark.task.resource.gpu.amount", 0.33) \
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


# Do a test connection to a DB and show the GPU in action
from pyspark.sql import functions as F

df = spark.read.table("DataLakeTable")
print(f"Columns: {len(df.columns)}")
print(f"Schema: {df.schema}")
# Look for 'Gpu' operators in the output
df.limit(5).explain(mode="formatted")


# In[11]:


# Access the Java 'SessionState' through the back door
jvm_session_state = spark._jsparkSession.sessionState()

# Check if the Catalyst Optimizer is using the RAPIDS extensions
pprint.pprint(f"Experimental Methods: {jvm_session_state.experimentalMethods()}")

# Access the experimental methods via the JVM bridge
experimental = spark._jsparkSession.sessionState().experimentalMethods()

# Enable SampleExec (another commonly disabled-by-default op)
spark.conf.set("spark.rapids.sql.exec.SampleExec", "true")
spark.conf.set("spark.sql.broadcastTimeout", "1200") # Increase to 20 mins


# In[ ]:


# Create two large-ish dataframes and join them
# This creates 10 million rows to give the GPU something to chew on
left_df = spark.range(0, 10000000) \
    .withColumn("join_key", F.col("id") % 1000) \
    .withColumn("data_value", F.rand(seed=42) * 100)

right_df = spark.range(0, 1000) \
    .withColumnRenamed("id", "join_key") \
    .withColumn("category", F.concat(F.lit("Category_"), F.col("join_key")))

# We use a inner join here on 'join_key'
# GPUs prefer HASH so we give it a hint
joined_df = left_df.hint("SHUFFLE_HASH").join(right_df, on="join_key", how="inner")

# Perform an Aggregation to trigger a Shuffle
final_result = joined_df.groupBy("category") \
    .agg(F.avg("data_value").alias("avg_val")) \
    .orderBy(F.desc("avg_val"))


final_result.show(10)

# Check the Physical Plan
# Look for 'GpuHashJoin' and 'GpuColumnarExchange'
final_result.explain(mode="formatted")


# In[ ]:


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


# In[ ]:


# Train the model
# This runs the training logic in C++ on the GPU via cuML
print("Training Spark RAPIDS ML model...")
rf_model = rf_classifier.fit(training_data)
print("Model training complete.")


# In[ ]:


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




# In[ ]:


import spark_rapids_ml.metrics.MulticlassMetrics as mm
#print(f"Available in metrics: {help(mm)}")


# In[ ]:


accuracy = predictions.filter("prediction = fraud_trx").count() / predictions.count()

print(f"GPU-Accelerated Accuracy: {accuracy:.4f}")


# In[18]:


#from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from spark_rapids_ml.metrics.MulticlassMetrics import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(
    labelCol="fraud_trx", 
    predictionCol="prediction", 
    metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy: {accuracy:.4f}")



# In[ ]:


from spark_rapids_ml.classification import RandomForestClassifier as cuRFC
from spark_rapids_ml.classification import RandomForestClassificationModel as cuRFCM
#print(f"Available in metrics: {help(cuRFCM)}")


# In[ ]:


#dir(rf_model)
# save to s3a
#rf_model.save("spark-rapids-ml/cuRFCM.out")


# In[ ]:


## https://medium.com/rapids-ai/rapids-forest-inference-library-prediction-at-100-million-rows-per-second-19558890bc35


# In[60]:


# 1. Check the internal __dict__ for anything containing 'json' or 'model'
internal_keys = [k for k in rf_model.__dict__.keys() if 'model' in k or 'json' in k or 'cuml' in k]
print(f"Potential internal keys: {internal_keys}")

import json

# 2. Try to extract from the most likely Spark RAPIDS internal locations
model_data = None

if hasattr(rf_model, "_model_json"):
    print("_model_json")
    json_model_data = rf_model._model_json
    #pprint.pprint(json_model_data)

    json_str = json.dumps(json_model_data) if isinstance(json_model_data, list) else json_model_data
    with open("fraud_rf_model.json", "w") as f:
        f.write(json_str)
if hasattr(rf_model, "_treelite_model"):
    print("_treelite_model") 
    treelite_model_data = rf_model._treelite_model
    #pprint.pprint(treelite_model_data)
    
    with open("fraud_rf_model.tl", "w") as f:
        f.write(treelite_model_data)
if "_rf_spark_model" in rf_model.__dict__:
    print("_rf_spark_model")    
    model_data = rf_model.__dict__["_rf_spark_model"]
    #pprint.pprint(model_data)        
if "_cuml_params" in rf_model.__dict__:
    print("_cuml_params")    
    model_data = rf_model.__dict__["_cuml_params"]
    #pprint.pprint(model_data)
#    json_str = json.dumps(model_data) if isinstance(model_data, list) else model_data
#    with open("fraud_rf_model_cuml.json", "w") as f:
#        f.write(json_str)



# In[92]:


import treelite
import cuml
import spark_rapids_ml
print(f"Treelite: {treelite.__version__}")
print(f"cuML:     {cuml.__version__}")
print(f"spark_rapids_ml:     {spark_rapids_ml.__version__}")


# In[93]:


# cuML v25.10.00 requires Treelite v4.4.1 but the serialized Treelite model it generates
# appears to be  v3.9
main_model = treelite.Model.deserialize("fraud_rf_model.tl")


# In[ ]:


# cuML v25.10.00 requires Treelite v4.4.1 but the serialized Treelite model it generates
# appears to be  v3.9


# In[ ]:





# In[ ]:


# --- STAGE 2: CLEAN LOAD ---
# Use the 'ignore unknown' flag to bypass the "threshold_type" error
tl_model = treelite.frontend.from_xgboost_json(
    "fraud_rf_model.json", 
    allow_unknown_field=True
)



# In[57]:


tl_model.set_metadata(
    num_feature=14,              # Schema requires feature count
    task_type='kBinaryClf',      # Schema requires task definition
    average_tree_output=True,    # Schema requires aggregation type (RF=True)
    num_target=1,
    num_class=[1]
)


# In[ ]:


# --- STAGE 3: METADATA STAMPING ---
# We must manually define the task so the GPU knows how to handle the trees
tl_model.set_metadata(
    num_feature=14,
    task_type='kBinaryClf',      # Binary classification for Fraud
    average_tree_output=True,    # This makes it a Random Forest (not GBT)
    num_target=1,
    num_class=[1]
)



# In[ ]:


# --- STAGE 4: GPU COMPILATION ---
# This is the "TensorRT-equivalent" step for Tree models
gpu_inference_engine = ForestInference.load_from_treelite_model(tl_model)

print("🚀 Optimization Complete. Model is live on GPU.")


# In[ ]:





# In[ ]:





# In[35]:


import treelite
from cuml.fil import ForestInference
import numpy as np

# 1. Load the trees (ignoring the "rich" Spark headers)
tl_model = treelite.frontend.load_xgboost_model(
    "fraud_rf_model.json", 
    allow_unknown_field=True
)

# 2. Manually initialize the missing metadata
# For your Random Forest Classifier:
# - num_feature: 14 (as seen in your metadata)
# - task_type: 'kBinaryClf' (since it's a fraud classifier)
# - num_class: [1] (for binary/single-target models, num_class is often [1])
tl_model.set_metadata(
    num_feature=14,
    task_type='kBinaryClf',
    average_tree_output=True,  # Crucial for Random Forests
    num_target=1,
    num_class=[1]
)

# 3. Load the model into the GPU FIL engine
# This is the "TensorRT-style" bare-metal optimization step
gpu_model = ForestInference.load_from_treelite_model(tl_model)

# 4. Verify with your 14 features
dummy_input = np.random.rand(1, 14).astype(np.float32)
preds = gpu_model.predict(dummy_input)

print("✅ Model successfully optimized and loaded into GPU!")
print(f"Sample Prediction: {preds}")


# In[29]:


from cuml.fil import ForestInference
import numpy as np

# Load into the optimized GPU engine
# This is the "TensorRT" equivalent for Tree models
gpu_model = ForestInference.load(
    "fraud_rf_model.json", 
    model_type="xgboost_json"
)


# In[22]:


import treelite

from cuml.fil import ForestInference
print("✅ Successfully imported from cuml.fil")


# In[28]:


import xgboost as xgb
from cuml.fil import ForestInference



gpu_inference_model = ForestInference.load_from_treelite_model(
    treelite_model_data
)



# In[ ]:


gpu_inference_model = ForestInference.load(treelite_model_data, model_type="treelite_checkpoint")

gpu_inference_model = ForestInference.load("fraud_rf_model.tl", model_type="treelite_checkpoint")


# In[21]:


# Assuming 'model' is your RandomForestClassificationModel instance
import json
params = rf_model.cuml_params

print("--- Underlying cuML Parameters ---")
print(json.dumps(params, indent=4))


# In[19]:


##SKIP

import json

# Extract the parameter map which stores the 'model_json' from the constructor
params = rf_model.extractParamMap()

# Search the params for the model data
model_data = None
for p, v in params.items():
    if "model_json" in p.name:
        model_data = v
        print("found it!")
        break

if model_data:
    # If it's a list, we need to join it into a single string
    final_json = model_data if isinstance(model_data, str) else json.dumps(model_data)
    
    with open("fraud_rf_model.json", "w") as f:
        f.write(final_json)
    print("✅ Success! Model extracted to fraud_rf_model.json")
else:
    print("❌ Could not find 'model_json' in params. Printing all keys to debug:")
    print([p.name for p in params.keys()])


# In[ ]:


from cuml.fil import ForestInference


# In[34]:


from cuml.fil import ForestInference
import numpy as np

# Load the model into the FIL engine
# This is the 'compilation' step that gives you TensorRT-like performance
fm = ForestInference.load("fraud_rf_model.json", model_type="xgboost_json")

# Prepare your 14 features as a float32 array
# This will run significantly faster than Spark's .transform()
test_input = np.random.rand(1, 14).astype(np.float32)
predictions = fm.predict(test_input)

print(f"Prediction: {predictions}")


# In[27]:


import treelite

# 1. Load the model from the JSON you just saved
model = treelite.Model.load("fraud_rf_model.json", model_format='xgboost_json')

# 2. Compile the model into a shared library (.so)
# This performs optimizations equivalent to TensorRT's layer fusion
model.export_lib(
    toolchain='gcc', 
    libpath='./fraud_rf_engine.so', 
    params={'parallel_comp': 4},
    verbose=True
)



# In[ ]:





# In[ ]:




