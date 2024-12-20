import os
import boto3

def upload_folder_to_s3(local_folder: str, bucket_name: str, s3_folder: str):
    """
    Uploads a folder and its contents to an S3 bucket.

    :param local_folder: Path to the local folder to upload.
    :param bucket_name: Name of the target S3 bucket.
    :param s3_folder: Folder in the S3 bucket where files will be uploaded.
    """
    # Initialize S3 client
    s3_client = boto3.client('s3')

    for root, dirs, files in os.walk(local_folder):
        for file in files:
            local_file_path = os.path.join(root, file)
            # Compute relative path for the S3 key
            relative_path = os.path.relpath(local_file_path, local_folder)
            s3_key = os.path.join(s3_folder, relative_path).replace("\\", "/")  # Ensure correct S3 path format
            
            try:
                print(f"Uploading {local_file_path} to s3://{bucket_name}/{s3_key}")
                s3_client.upload_file(local_file_path, bucket_name, s3_key)
            except Exception as e:
                print(f"Failed to upload {local_file_path}: {e}")

# Example usage
if __name__ == "__main__":
    LOCAL_FOLDER = "/path/to/local/folder"
    BUCKET_NAME = "your-s3-bucket-name"
    S3_FOLDER = "your-s3-folder"

    upload_folder_to_s3(LOCAL_FOLDER, BUCKET_NAME, S3_FOLDER)
