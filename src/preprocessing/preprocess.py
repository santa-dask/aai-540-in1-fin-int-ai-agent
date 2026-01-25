
import sys
import os

# Add the 'src' directory to the system path to allow for absolute imports.
# This is necessary so that this script can find the 'utils' module.
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from google.api_core import exceptions
from utils import config_loader as cl

import bigframes.pandas as bpd

def preprocess_raw_to_staging(project_id, dataset_id, bucket_name):
    # Initialize BigFrames options
    bpd.options.bigquery.project = project_id
    bpd.options.bigquery.location = "US"

    # 1. Point to the Raw CSV in the Data Lake (Bronze Layer)
    gcs_uri = f"gs://{bucket_name}/raw/complaints*.csv"
    
    # Create a BigQuery DataFrame directly from the GCS external file
    # This acts as an 'External Table'—no data is moved yet.
    df = bpd.read_csv(gcs_uri)

    # 2. Preprocessing Work (Pandas-style)
    # Example: Selecting relevant columns and handling nulls
    columns_to_keep = ['date_received', 'product', 'issue', 'complaint_what_happened', 'zip_code']
    df_clean = df[columns_to_keep].dropna(subset=['complaint_what_happened'])

    # Example: De-identification 
    # Redact ZIP codes to prevent them acting as geographic proxies
    df_clean['zip_code'] = df_clean['zip_code'].str.slice(0, 3) + "XX"

    # 3. Save to the Staging Table
    # This materializes the preprocessed data into a BigQuery native table
    staging_table_id = f"{project_id}.{dataset_id}.stg_complaints"
    df_clean.to_gbq(staging_table_id, if_exists="replace")
    
    print(f"✅ Preprocessing complete. Staging table created: {staging_table_id}")

if __name__ == "__main__":
    PROJECT = cl.config_loader.get(cl.PROJECT_ID)
    RAW_BUCKET = f"{PROJECT}-{cl.config_loader.get(cl.RAW_BUCKET)}"
    preprocess_raw_to_staging(PROJECT, "cfpb_analysis", RAW_BUCKET)