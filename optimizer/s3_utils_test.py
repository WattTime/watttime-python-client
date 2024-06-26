

# https://towardsdatascience.com/reading-and-writing-files-from-to-amazon-s3-with-pandas-ccaf90bfe86c

import optimizer.s3 as s3u 
import pandas as pd
import os

df = pd.read_csv("/home/jennifer.badolato/watttime-python-client-aer-algo/optimizer/us_region_meta_data.csv")

s3 = s3u.s3_utils()
key = 'meta_data_test.csv'

s3.store_csvdataframe(dataframe=df, key=key)

AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")

books_df = pd.read_csv(
    f"s3://{AWS_S3_BUCKET}/{key}",
    storage_options={
        "key": AWS_ACCESS_KEY,
        "secret": AWS_SECRET_ACCESS_KEY
    },
)

print(books_df.head())