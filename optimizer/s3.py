"""
Util functions for S3.
Expand as needed.
"""

import boto3
import pandas as pd
import os
import io
import json

AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")


class s3_utils:
    """
    Utilities to IO operations on Amazon S3.
    This library assumes that credentials are stored locally.
    """

    def __init__(self, prod=False):
        """
        Connect to S3 when object is created
        """
        self.S3 = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )

    def load_file(self, file: str):
        """
        Args:
            file: Filename, i.e. s3 Key

        Returns:
            json body

        Example:
            s3.load_file("test.csv")

        """

        response = self.S3.get_object(Bucket=AWS_S3_BUCKET, Key=file)
        return response["Body"].read()

    def store_file(self, filesource: str, filedestination: str):  # type: ignore
        """
        Args:
            filesource: Filename with path and extension
            filedestination: filename with path and extension

        Returns:
            Not defined yet

        Example:
            s3.store_file("learn.txt","learn.txt")

        """
        self.S3.upload_file(
            Filename=filesource, Bucket=AWS_S3_BUCKET, Key=filedestination
        )

    def load_csvdataframe(self, file: str, bucket=AWS_S3_BUCKET):
        """
        Args:
            file: Filename, i.e. s3 Key

        Returns:
            pd.DataFrame

        Example:
            s3.load_csvdataframe("test.csv")

        """
        data = self.load_file(file=file)
        return pd.read_csv(io.BytesIO(data))

    def store_csvdataframe(
        self, dataframe: pd.DataFrame, file: str, bucket=AWS_S3_BUCKET
    ):
        """
        Args:
            dataframe: Pandas dataframe
            file: filename, i.e. "files/books.csv"
            bucket: bucketname
        Returns:
            response status code
        Example:
            s3.store_csvdataframe(dataframe,"files/data.csv")
        """

        csv_buffer = io.StringIO()
        dataframe.to_csv(csv_buffer, index=False)
        response = self.S3.put_object(
            Bucket=bucket, Key=file, Body=csv_buffer.getvalue()
        )
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

        if status == 200:
            print(f"Successful S3 put_object response. Status - {status}")
        else:
            print(f"Unsuccessful S3 put_object response. Status - {status}")

    def store_dictionary(
        self, dictionary, file: str, bucket=AWS_S3_BUCKET
    ):
        response = self.S3.put_object(
            Body=dictionary,
            Bucket=bucket,
            Key=file
        )
        
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

        if status == 200:
            print(f"Successful S3 put_object response. Status - {status}")
        else:
            print(f"Unsuccessful S3 put_object response. Status - {status}")