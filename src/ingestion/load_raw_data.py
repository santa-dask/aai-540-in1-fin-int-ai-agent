import os
import sys

# Add the 'src' directory to the system path to allow for absolute imports.
# This is necessary so that this script can find the 'utils' module.
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from google.cloud import storage
from google.api_core import exceptions
from util.config_loader import config_loader

def load_to_raw_bucket(project_id, bucket_name, local_file_path):
    """
    Uploads the local CSV file to the 'raw/' folder in the GCS Data Lake.
    """
    # 1. Initialize the GCS Client
    client = storage.Client(project=project_id)
    
    try:
        bucket = client.get_bucket(bucket_name)
    except Exception as e:
        print(f"Bucket {bucket_name} not found. Ensure infrastructure setup is run. Error: {e}")
        return

    # 2. Define the target 'folder' path
    # In GCS, folders are simulated by prefixes in the name.
    blob_name = config_loader.get("data.raw_file_path")
    blob = bucket.blob(blob_name)

    # 3. Perform the upload
    print(f"Uploading {local_file_path} to GCS bucket...")
    
    # upload_from_filename is optimized for large file transfers
    blob.upload_from_filename(f"{local_file_path}")

    print(f"Success! File available in GCS")

if __name__ == "__main__":
    # Ensure your environment variable is set
    PROJECT = config_loader.get("PROJECT_ID")
    RAW_BUCKET = config_loader.get("GCS_BUCKET")
    RAW_FILE_PATH =  f"{os.getenv('PROJECT_DIR')}/data/{config_loader.get('data.raw_file_path').split('/')[-1]}" # Ensure file exists here

    if os.path.exists(RAW_FILE_PATH):
        load_to_raw_bucket(PROJECT, RAW_BUCKET, "../data/complaints1.csv")
    else:
        print(f" Error: {RAW_FILE_PATH} not found. Please place your dataset in the data/ directory.")
    
