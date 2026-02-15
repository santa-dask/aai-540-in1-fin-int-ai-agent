
import sys
import os

# Add the 'src' directory to the system path to allow for absolute imports.
# This is necessary so that this script can find the 'utils' module.
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from google.api_core import exceptions
from google.cloud import bigquery
from google.cloud import storage
from utils import config_loader as cl

import pandas as pd
import io


def get_cfpb_schema():
    """
    Define explicit schema for CFPB complaints dataset.
    Uses valid BigQuery column names (no special characters like ?, spaces allowed with backticks).
    
    Returns:
        list: List of google.cloud.bigquery.SchemaField objects
    """
    schema = [
        bigquery.SchemaField("Date received", "DATE", mode="NULLABLE"),
        bigquery.SchemaField("Product", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Sub-product", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Issue", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Sub-issue", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Consumer complaint narrative", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Company public response", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Company", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("State", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("ZIP code", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Tags", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Consumer consent provided", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Submitted via", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Date sent to company", "DATE", mode="NULLABLE"),
        bigquery.SchemaField("Company response to consumer", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Timely response", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Consumer disputed", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Complaint ID", "INT64", mode="NULLABLE"),
    ]
    return schema

def preprocess_raw_to_staging(project_id, dataset_id, bucket_name):
    """
    Reads raw data from GCS (gzip or CSV), applies explicit schema, and writes to BigQuery staging table.
    Uses pandas to handle CSV column mapping and BigQuery Load Job API for explicit schema enforcement.
    Supports both .csv.gz and .csv formats.
    """
    
    # 1. Point to the Raw CSV in the Data Lake (Bronze Layer)
    # Try both gzip and plain CSV formats
    gcs_uri_gz = f"gs://{bucket_name}/raw/complaints.csv.gz"
    gcs_uri_csv = f"gs://{bucket_name}/raw/complaints.csv"
    
    # Determine which file exists by checking GCS
    storage_client = storage.Client(project=project_id)
    bucket_obj = storage_client.bucket(bucket_name)
    
    gcs_uri = None
    for uri in [gcs_uri_gz, gcs_uri_csv]:
        blob_name = uri.replace(f"gs://{bucket_name}/", "")
        blob = bucket_obj.blob(blob_name)
        try:
            blob.reload()  # Check if blob exists
            gcs_uri = uri
            print(f"✓ Found file: {uri}")
            break
        except exceptions.NotFound:
            print(f"File not found: {uri}")
            continue
    
    if gcs_uri is None:
        print(f"No valid raw file found in gs://{bucket_name}/raw/")
        return
    
    print(f"GCS RAW File Location: {gcs_uri}")
    
    # 2. Read CSV from GCS with pandas
    # This allows us to handle special characters in column names
    print("Reading CSV from GCS...")
    df = pd.read_csv(gcs_uri)
    print(f"✓ Loaded {len(df):,} rows from CSV")
    print(f"  Columns: {list(df.columns)}")
    
    # 3. Rename columns to valid BigQuery names
    # Replace special characters (like ?) with valid names
    column_name_mapping = {
        "Consumer consent provided?": "Consumer consent provided",
        "Timely response?": "Timely response",
        "Consumer disputed?": "Consumer disputed",
    }
    df = df.rename(columns=column_name_mapping)
    print(f"✓ Renamed {len(column_name_mapping)} columns for BigQuery compatibility")
    
    # 4. Ensure proper data types
    # Parse dates as DATE type
    date_columns = ["Date received", "Date sent to company"]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.date
    
    # Ensure Complaint ID is integer
    if "Complaint ID" in df.columns:
        df["Complaint ID"] = pd.to_numeric(df["Complaint ID"], errors='coerce')
    
    print(f"✓ Converted columns to specified data types")
    
    # 5. Write to BigQuery with explicit schema
    staging_table_id = f"{project_id}.{dataset_id}.stg_complaints"
    print(f"\nWriting to BigQuery staging table: {staging_table_id}")
    
    # Initialize BigQuery client
    client = bigquery.Client(project=project_id)
    
    # Configure job with explicit schema
    schema = get_cfpb_schema()
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        autodetect=False,  # Do not auto-detect; use explicit schema only
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,  # Replace existing table
    )
    
    # Load pandas DataFrame to BigQuery
    print("Uploading data to BigQuery...")
    job = client.load_table_from_dataframe(
        df,
        staging_table_id,
        job_config=job_config,
        location="US",
    )
    job.result()  # Wait for job to complete
    
    # Verify load results
    destination_table = client.get_table(staging_table_id)
    print(f"✓ Loaded {destination_table.num_rows:,} rows into {staging_table_id}")
    print(f"✓ Preprocessing complete. Staging table created with explicit schema.")

if __name__ == "__main__":
    PROJECT = cl.config_loader.get(cl.PROJECT_ID)
    RAW_BUCKET = f"{PROJECT}-{cl.config_loader.get(cl.RAW_BUCKET)}"
    preprocess_raw_to_staging(PROJECT, "cfpb_analysis", RAW_BUCKET)