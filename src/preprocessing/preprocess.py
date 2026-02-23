
import sys
import os
import pandas as pd

# Add the 'src' directory to the system path to allow for absolute imports.
# This is necessary so that this script can find the 'utils' module.
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from google.api_core import exceptions
from util.config_loader import config_loader

import bigframes.pandas as bpd

def preprocess_raw_to_staging(project_id, dataset_id, bucket_name):
    # Initialize BigFrames options
    bpd.options.bigquery.project = project_id
    bpd.options.bigquery.location = config_loader.get("project.region")

    # 1. Point to the Raw CSV in the Data Lake (Bronze Layer)
    gcs_uri = f"gs://{bucket_name}/{config_loader.get('data.raw_file_path')}"
    #print(f" GCS RAW File Location: {gcs_uri}")
    
    # Create a BigQuery DataFrame directly from the GCS external file
    # This acts as an 'External Table'—no data is moved yet.
    df = bpd.read_csv(gcs_uri)

    # 2. Preprocessing Work (Pandas-style)
    # Example: Selecting relevant columns and handling nulls
    #columns_to_keep = ['date_received', 'product', 'issue', 'complaint_what_happened', 'zip_code']
    #df_clean = df[columns_to_keep].dropna(subset=['complaint_what_happened'])
    df_clean = df
    # Example: De-identification 
    # Redact ZIP codes to prevent them acting as geographic proxies
    #df_clean['zip_code'] = df_clean['zip_code'].str.slice(0, 3) + "XX"

    # 3. Save to the Staging Table
    # This materializes the preprocessed data into a BigQuery native table
    staging_table_id = f"{project_id}.{dataset_id}.stg_complaints"
    df_clean.to_gbq(staging_table_id, if_exists="replace")
    
    print(f"Preprocessing complete. Staging table created: {dataset_id}.stg_complaints")

def preprocess_data():
    gcs_uri = f"gs://{config_loader.get('GCS_BUCKET')}/{config_loader.get('data.raw_file_path')}"
    complaints_ds = pd.read_csv(gcs_uri)
    print(f"Number of records before removing records with Nan value : {len(complaints_ds)}")
    complaints_ds = complaints_ds.dropna(subset=['Consumer complaint narrative'])
    complaints_ds = complaints_ds.reset_index(drop=True)

    long_complaints_ds = complaints_ds[complaints_ds['Consumer complaint narrative'].str.len() > 600].copy()
    long_complaints_ds = long_complaints_ds.reset_index(drop=True)
    return long_complaints_ds

def split_data( long_complaints_ds, num_of_records = 100000):
    training_ds = long_complaints_ds[:num_of_records]
    testing_ds = long_complaints_ds[num_of_records: int(num_of_records * 1.20)]
    sample_ds = long_complaints_ds[int(num_of_records*1.2):int(num_of_records*1.21)]

    # Get the project directory, assuming it's correctly set as an environment variable
    project_dir = os.getenv('PROJECT_DIR', '/content/aai-540-in1-fin-int-ai-agent')

    # Construct the path to the 'data' directory
    data_dir = os.path.join(project_dir, 'data')

    # Create the 'data' directory if it does not exist
    if not os.path.exists(data_dir):
      os.makedirs(data_dir)
      print(f"Created directory: {data_dir}")

    training_ds.to_csv(os.path.join(project_dir, config_loader.get("data.training_file_path")))
    testing_ds.to_csv(os.path.join(project_dir,config_loader.get("data.testing_file_path")))
    sample_ds.to_csv(os.path.join(project_dir,config_loader.get("data.sample_file_path")))

if __name__ == "__main__":
    PROJECT = config_loader.get("PROJECT_ID")
    RAW_BUCKET = config_loader.get("GCS_BUCKET")
    preprocess_raw_to_staging(PROJECT, "cfpb_analysis", RAW_BUCKET)