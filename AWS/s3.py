import os
import pprint
import random
import string
from urllib import response
import boto3

def main():

    print("📚 Setting up the environment...")

    pp = pprint.PrettyPrinter(indent=2)

    os.environ["AWS_SHARED_CREDENTIALS_FILE"] = os.path.abspath(".aws/credentials")
    os.environ["AWS_CONFIG_FILE"] = os.path.abspath(".aws/config")

    s3 = boto3.client("s3", region_name="eu-west-1")
    print("✅ Environment setup complete!")
    print(f"🌍 Using AWS region: {s3.meta.region_name}")

    print("📋 Listing all S3 buckets in your account...")
    response = s3.list_buckets()

    print("\n📦 Raw response from AWS:")
    pp.pprint(response)

    print("\n📦 Your current S3 buckets:")
    if response["Buckets"]:
        for bucket in response["Buckets"]:
            print(f"- {bucket['Name']}")
    else:
        print("No buckets found in your account")

    print(f"\n✅ Successfully retrieved {len(response['Buckets'])} buckets")

    bucket_name = "amzn-main-bucket-3"
    print(f"⬆️  Uploading file to bucket: {bucket_name}")

    try:
        s3.upload_file("Data/data_for_analysis/cleaned_data.parquet", bucket_name, "cleaned_data.parquet")
        print("✅ Upload successful!")
        print(f"📍 File location: s3://{bucket_name}/cleaned_data.parquet")
        print(f"ℹ️ Note: The https URL https://{bucket_name}.s3.eu-west-1.amazonaws.com/cleaned_data.parquet")
        print("   won't work directly because S3 objects are private by default!")
        print("   We'll generate a pre-signed URL later to access this file via HTTPS.")

        # Verify the upload by listing objects in the bucket
        objects = s3.list_objects_v2(Bucket=bucket_name)
        print("\n📦 Current bucket contents:")
        for obj in objects.get("Contents", []):
            print(f"- {obj['Key']} ({obj['Size']} bytes)")
    except Exception as e:
        print(f"❌ Error uploading file: {str(e)}")


if __name__ == "__main__":
    main()