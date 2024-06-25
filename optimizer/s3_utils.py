

# https://towardsdatascience.com/reading-and-writing-files-from-to-amazon-s3-with-pandas-ccaf90bfe86c





import importlib
import optimizer.s3 as s3u 
import pandas as pd

df = pd.read_csv("/home/jennifer.badolato/watttime-python-client-aer-algo/optimizer/us_region_meta_data.csv")

s3 = s3u.s3_utils()

s3.store_csvdataframe(dataframe=df, key='meta_data_test/csv')



AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")
KEY = "hello-from-cindy.txt"


s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

client.list_objects(
    Bucket='dataclinic-watttime',
    MaxKeys=2,
)


books_df = pd.read_csv(
    f"s3://{AWS_S3_BUCKET}/{key}",
    storage_options={
        "key": AWS_ACCESS_KEY_ID,
        "secret": AWS_SECRET_ACCESS_KEY,
        "token": AWS_SESSION_TOKEN,
    },
)

key = "hello-from-cindy.txt"

books_df = pd.read_csv(
    f"s3://{AWS_S3_BUCKET}/{key}",
    storage_options={
        "key": AWS_ACCESS_KEY,
        "secret": AWS_SECRET_ACCESS_KEY
    },
)