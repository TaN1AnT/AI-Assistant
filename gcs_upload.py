import os
from google.cloud import storage
from dotenv import load_dotenv
load_dotenv()

# --- CONFIGURATION ---
# 1. Path to your service account JSON file
SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# 2. Your GCS Bucket URL (e.g., gs://my-bucket-name/optional-folder/)
GCS_BUCKET_URL = os.getenv("GCS_BUCKET_URL")

# 3. Local path to the file or folder you want to upload
LOCAL_PATH = "D:\AI_Orchestration\documents"


def upload_to_gcs(json_key, bucket_url, local_path):
    """Uploads a file or a whole directory to a GCS bucket."""
    
    # Initialize the client using the service account key
    client = storage.Client.from_service_account_json(json_key)
    
    # Parse the bucket name and destination prefix from the URL
    # Removes 'gs://' and splits the bucket name from the path
    clean_url = bucket_url.replace("gs://", "")
    parts = clean_url.split("/", 1)
    bucket_name = parts[0]
    dest_prefix = parts[1] if len(parts) > 1 else ""

    bucket = client.bucket(bucket_name)

    # Check if local path is a file or a directory
    if os.path.isfile(local_path):
        # --- SINGLE FILE UPLOAD ---
        file_name = os.path.basename(local_path)
        blob_path = os.path.join(dest_prefix, file_name).replace("\\", "/")
        blob = bucket.blob(blob_path)
        
        print(f"Uploading {file_name} to gs://{bucket_name}/{blob_path}...")
        blob.upload_from_filename(local_path)
        print("Upload complete!")

    elif os.path.isdir(local_path):
        # --- DIRECTORY UPLOAD ---
        print(f"Uploading folder {local_path}...")
        for root, dirs, files in os.walk(local_path):
            for file in files:
                local_file = os.path.join(root, file)
                
                # Create a relative path for GCS
                rel_path = os.relpath(local_file, local_path)
                blob_path = os.path.join(dest_prefix, rel_path).replace("\\", "/")
                
                blob = bucket.blob(blob_path)
                print(f"  -> Uploading {rel_path}...")
                blob.upload_from_filename(local_file)
        print("Folder upload complete!")
    else:
        print(f"Error: {local_path} is not a valid file or directory.")

if __name__ == "__main__":
    upload_to_gcs(SERVICE_ACCOUNT_JSON, GCS_BUCKET_URL, LOCAL_PATH)