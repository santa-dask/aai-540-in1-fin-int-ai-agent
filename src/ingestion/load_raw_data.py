import os
import sys

# Add the 'src' directory to the system path to allow for absolute imports.
# This is necessary so that this script can find the 'utils' module.
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from google.cloud import storage
from google.api_core import exceptions
from utils import config_loader as cl

def load_to_raw_bucket(project_id, bucket_name, local_file_path):
    """
    Uploads the local gzip CSV file to the 'raw/' folder in the GCS Data Lake.
    Supports both .csv and .csv.gz files.
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
    blob_name = f"raw/{os.path.basename(local_file_path)}"
    blob = bucket.blob(blob_name)

    # 3. Check if file exists
    if not os.path.exists(local_file_path):
        print(f"Error: File not found at {local_file_path}")
        return

    # 4. Perform the upload
    print(f"⏳ Uploading {local_file_path} to gs://{bucket_name}/{blob_name}...")
    print(f"   File size: {os.path.getsize(local_file_path) / (1024*1024):.2f} MB")
    
    # upload_from_filename is optimized for large file transfers
    blob.upload_from_filename(local_file_path)

    print(f"✅ Success! File available at: gs://{bucket_name}/{blob_name}")

if __name__ == "__main__":
    # Ensure your environment variable is set
    PROJECT = cl.config_loader.get(cl.PROJECT_ID)
    RAW_BUCKET = f"{PROJECT}-{cl.config_loader.get(cl.RAW_BUCKET)}"
    RAW_FILE_PATH = cl.config_loader.get(cl.RAW_FILE_PATH)
    
    # Calculate the actual path from project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    full_file_path = os.path.join(project_root, RAW_FILE_PATH)

    if os.path.exists(full_file_path):
        load_to_raw_bucket(PROJECT, RAW_BUCKET, full_file_path)
    else:
        print(f"Error: {full_file_path} not found. Please place your dataset in the data/ directory.")
    
