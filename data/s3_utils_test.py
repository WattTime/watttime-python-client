# https://towardsdatascience.com/reading-and-writing-files-from-to-amazon-s3-with-pandas-ccaf90bfe86c

import data.s3 as s3u
import pandas as pd
import os

AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

s3 = s3u.s3_utils()
key = "PJM_NJ_co2_moer_2024-01-01_2024-02-01.csv"
# print(s3.store_csvdataframe(file=key).head())

"""

s3.store_file(
    filesource="",
    filedestination="PJM_NJ_co2_moer_2024-01-01_2024-02-01.csv"
)

json_ = s3.load_file(
    "PJM_NJ_co2_moer_2024-01-01_2024-02-01.csv"
)

books_df = pd.read_csv(
    f"s3://{AWS_S3_BUCKET}/PJM_NJ_co2_moer_2024-01-01_2024-02-01.csv",
    storage_options={
        "key": AWS_ACCESS_KEY,
        "secret": AWS_SECRET_ACCESS_KEY
    },
)

print(books_df.head())

"""
