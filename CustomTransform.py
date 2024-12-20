import os
import boto3

def upload_directory(local_directory, bucket_name, s3_prefix=''):
    s3 = boto3.client('s3')

    # Walk through the local directory
    for root, dirs, files in os.walk(local_directory):
        for file in files:
            # Get the full local file path
            local_file_path = os.path.join(root, file)
            
            # Create the corresponding S3 key (folder structure)
            relative_path = os.path.relpath(local_file_path, local_directory)
            s3_key = os.path.join(s3_prefix, relative_path)

            # Upload the file to S3
            s3.upload_file(local_file_path, bucket_name, s3_key)
            print(f"Uploaded {local_file_path} to s3://{bucket_name}/{s3_key}")

# Usage
local_folder = 'path/to/your/local/folder'  # Local folder you want to upload
bucket_name = 'your-bucket-name'            # Your S3 bucket name
s3_prefix = 'your/desired/folder/prefix'    # Optional: Prefix to organize files in S3

upload_directory(local_folder, bucket_name, s3_prefix)
