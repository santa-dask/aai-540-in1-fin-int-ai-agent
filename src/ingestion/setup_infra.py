import sys
import os

# Add the 'src' directory to the system path to allow for absolute imports.
# This is necessary so that this script can find the 'utils' module.
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from google.cloud import storage, bigquery
from google.api_core import exceptions
from google.auth.exceptions import DefaultCredentialsError
from util.config_loader import config_loader

def create_gcs_buckets(project_id, location="US"):
    """
    Creates the required GCS buckets for the project lifecycle.
    It reads the bucket names from the configuration file.
    """
    try:
        client = storage.Client(project=project_id)
    except DefaultCredentialsError as exc:
        print(" Credentials not found.")
        
        # Debugging Service Account Key issues
        sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if sa_path:
            print(f"   - GOOGLE_APPLICATION_CREDENTIALS found: {sa_path}")
            if not os.path.exists(sa_path):
                print(f"   -  Error: The file '{sa_path}' does not exist. Check the path.")
        else:
            print("   - GOOGLE_APPLICATION_CREDENTIALS environment variable is NOT set.")

        print("   - If in Colab: Run 'from google.colab import auth; auth.authenticate_user()'")
        print("   - If local: Run 'gcloud auth application-default login'")
        print(f"   - Error: {exc}")
        return
    except Exception as e:
        print(f" Failed to initialize GCS Client: {e}")
        return
    
    # Use the constants from the config_loader to get the keys
    bucket_keys = [
        "GCS_BUCKET"
    ]
    
    bucket_names = [os.getenv(key) for key in bucket_keys]

    for bucket_name in bucket_names:
        if not bucket_name:
            print(f" Warning: A bucket name is not configured in YAML. Skipping.")
            continue
            
        full_name = f"{project_id}-{bucket_name}"
        full_name=f"{bucket_name}"
        bucket = client.bucket(full_name)
        
        try:
            if not bucket.exists():
                new_bucket = client.create_bucket(bucket, location=location)
                print(f" Created GCS Bucket: {new_bucket.name}")
            else:
                print(f" GCS Bucket already exists: {full_name}")
        except exceptions.Forbidden as e:
            print(f" GCS Permission Denied for {full_name}: {e}")
        except Exception as e:
            print(f" Failed to create GCS bucket {full_name}: {e}")

def create_bigquery_dataset(project_id, location="us-central1"):
    """
    Creates the BigQuery dataset if it does not already exist.
    It reads the dataset name from the configuration file.
    """
    try:
        client = bigquery.Client(project=project_id)
    except DefaultCredentialsError as exc:
        print(" Credentials not found.")
        
        sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if sa_path:
            print(f"   - GOOGLE_APPLICATION_CREDENTIALS found: {sa_path}")
            if not os.path.exists(sa_path):
                print(f"   -  Error: The file '{sa_path}' does not exist.")

        print("   - If in Colab: Run 'from google.colab import auth; auth.authenticate_user()'")
        print("   - If local: Run 'gcloud auth application-default login'")
        print(f"   - Error: {exc}")
        return
    except Exception as e:
        print(f" Failed to initialize BigQuery Client: {e}")
        return
    
    dataset_name = config_loader.get("bigquery.dataset")
    if not dataset_name:
        print(" Warning: 'bigquery.dataset' not configured in YAML. Skipping BigQuery setup.")
        return

    dataset_id = f"{project_id}.{dataset_name}"
    
    try:
        client.get_dataset(dataset_id)  # Make an API request.
        print(f"BigQuery Dataset already exists: {dataset_name}")
    except exceptions.NotFound:
        print(f"BigQuery Dataset not found: {dataset_id}. Creating...")
        try:
            dataset = bigquery.Dataset(dataset_id)
            dataset.location = location
            created_dataset = client.create_dataset(dataset, timeout=30)
            print(f" Created BigQuery Dataset: {dataset_name}")
        except exceptions.Forbidden as e:
            print(f" BigQuery Permission Denied for {dataset_name}: {e}")
        except Exception as e:
            print(f" Failed to create BigQuery dataset {dataset_name}: {e}")

def  setup_data_lake(project_id, location="us-central1"):
    create_gcs_buckets(project_id, location=location)
    create_bigquery_dataset(project_id, location=location)

if __name__ == "__main__":
    
    project_id = config_loader.get("PROJECT_ID")
    region = config_loader.get("project.region")

    if not project_id:
        print(" Error: Project ID not found. Check GOOGLE_CLOUD_PROJECT env var or config.yaml.")
    else:
        print(f"--- Setting up infrastructure for Project: {project_id} in {region} ---")
        create_gcs_buckets(project_id, location=region)
        create_bigquery_dataset(project_id, location=region)
        print("--- Infrastructure setup complete. ---")
