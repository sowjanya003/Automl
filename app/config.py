"""Configuration and environment variables"""
import os
from dotenv import load_dotenv

load_dotenv()

# Azure Cosmos DB
COSMOS_CONNECTION_STRING = os.getenv("COSMOS_CONNECTION_STRING")
DATABASE_NAME = os.getenv('DATABASE_NAME')

CONTAINER_NAME = os.getenv('USER_TABLE', 'UserData')
CONTAINER_CHAT_HISTORY = os.getenv('CHAT_HISTORY', 'ChatHistory')
THREAD_METADATA = os.getenv('THREAD_METADATA', 'ThreadMetadata')
DATASET_METADATA = os.getenv('DATASET_METADATA', 'DatasetMetadata')
MODEL_REGISTRY = os.getenv('MODEL_REGISTRY', 'ModelRegistry')
TEST_METRICS = os.getenv('TEST_METRICS', 'TestMetrics')

# Azure Blob Storage
BLOB_CONNECTION_STRING = os.getenv("BLOB_CONNECTION_STRING")
UPLOAD_CONTAINER = os.getenv("UPLOAD_CONTAINER", "uploaded-file")

# Azure AI
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP")
AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
AZURE_PROJECT_NAME = os.getenv("AZURE_PROJECT_NAME")
AGENT_MODEL = os.getenv("MODEL_DEPLOYMENT_NAME")

# Azure Functions
# Azure Functions
FUN1_URL = os.getenv("FUN1_URL")
FUN2_URL = os.getenv("FUN2_URL")
FUN3_URL = os.getenv("FUN3_URL")

# Databricks (Unity Catalog Volumes)
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
DATABRICKS_CLUSTER_ID = os.getenv("DATABRICKS_CLUSTER_ID")

# Root Volume path for AutoML-generated outputs (predictions, and later model
# artifacts if that gets added). Deliberately a SEPARATE catalog/schema from
# wherever source datasets are read (e.g. veriton-db.landing.datasets) so
# generated outputs have their own governance boundary, independent of the
# access rules on the raw landing data.
# Expected shape: /Volumes/<output_catalog>/<output_schema>/<output_volume>
DATABRICKS_OUTPUT_ROOT = os.getenv("DATABRICKS_OUTPUT_ROOT", "/Volumes/veriton-db/automl-db/users")

# Snowflake (named internal stages)
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")
SNOWFLAKE_ROLE = os.getenv("SNOWFLAKE_ROLE")  # optional

# Root stage reference for AutoML-generated outputs (predictions, and later
# model artifacts). Deliberately a SEPARATE schema from wherever source
# datasets are read (e.g. VERITON_DB.DATASETS.DATASETS_STAGE) so generated
# outputs have their own governance boundary, independent of the access
# rules on the raw landing data. Mirrors DATABRICKS_OUTPUT_ROOT's shape -
# one combined URI string, not separate database/schema/stage env vars.
# Expected shape: snowflake://stage/<DATABASE>/<SCHEMA>/<STAGE>
SNOWFLAKE_OUTPUT_ROOT = os.getenv("SNOWFLAKE_OUTPUT_ROOT")

# OneLake (Fabric lakehouse) - reuses the same VERITAS_* credentials and
# the same hardcoded workspace/lakehouse ("agenticBI" / "newagenticBI")
# already used for reading OneLake source files (see app/source_fetch.py).
# Unlike Databricks/Snowflake, there's no separate catalog/database to
# point at here, so the "output root" is just a dedicated folder inside
# that same lakehouse's Files/ area, keeping generated outputs apart from
# wherever source datasets are read from.
# Expected shape: Files/<subfolder>
ONELAKE_OUTPUT_ROOT = os.getenv("ONELAKE_OUTPUT_ROOT", "Files/automl-outputs")

# ML Task & Model Configurations
TASKS = ["classification", "regression", "forecasting", "multistep_forecasting", "clustering", "anomaly_detection"]

MODELS = {
    "classification": ["logistic_regression", "random_forest", "gradient_boosting", "xgboost"],
    "regression": ["ridge", "random_forest", "gradient_boosting", "xgboost"],
    "forecasting": ["arima", "prophet", "xgboost", "lightgbm", "catboost"],
    "multistep_forecasting": ["xgboost", "lightgbm", "catboost"],
    "clustering": ["kmeans", "kmeans_plusplus", "dbscan", "gmm"],
    "anomaly_detection": ["isolation_forest_fast", "isolation_forest_precise", "one_class_svm", 
                          "local_outlier_factor", "elliptic_envelope"]
}

METRICS = {
    "classification": ["accuracy", "f1", "precision", "recall", "roc_auc"],
    "regression": ["rmse", "mae", "r2", "mape"],
    "forecasting": ["rmse", "mae", "r2"],
    "multistep_forecasting": ["rmse", "mae", "r2", "mape"],
    "clustering": ["silhouette_score", "davies_bouldin_score", "calinski_harabasz"],
    "anomaly_detection": ["anomaly_score", "precision", "recall", "f1"]
}

DEFAULT_METRICS = {
    "classification": "f1",
    "regression": "rmse",
    "forecasting": "rmse",
    "multistep_forecasting": "rmse",
    "clustering": "silhouette_score",
    "anomaly_detection": "anomaly_score"
}

# ---------------------------------------------------------------------------
# Azure Function call timeouts (seconds)
#
# host.json sets functionTimeout: 00:10:00, so any client timeout below 600
# abandons runs the function is still executing. That is what produced
# "Read timed out (read timeout=180)" on a job FUN1 finished in 196s and
# reported as failed to the user even though the model was trained and saved.
#
# 660 = 600s function budget + 60s slack for upload and response.
# ---------------------------------------------------------------------------
FUN1_TIMEOUT = int(os.getenv("FUN1_TIMEOUT", "660"))   # training
FUN2_TIMEOUT = int(os.getenv("FUN2_TIMEOUT", "300"))   # batch inference
FUN3_TIMEOUT = int(os.getenv("FUN3_TIMEOUT", "120"))   # single prediction