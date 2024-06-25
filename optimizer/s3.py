"""
Util functions for S3
"""

import boto3
import joblib
import pandas as pd
import os
import io

AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")


class s3_utils:
    """
    Utilities to IO operations on Amazon S3.
    This library assumes that credentials are stored locally or class is being called from lambda.
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
        response = self.S3.get_object(Bucket=AWS_S3_BUCKET, Key=file)
        return response["Body"].read()

    def store_file(self, filesource: str, filedestination: str, bucket: AWS_S3_BUCKET):
        """
        Args:
            filesource: Filename with path and extension
            filedestination: filename with path and extension
            bucket: bucketname

        Returns:
            Not defined yet

        Example:
            s3.store_file("learn.txt","learn.txt")

        """
        self.S3.upload_file(filesource, bucket, filedestination)

    def delete_file(self, file: str, bucket=AWS_S3_BUCKET):
        """
        Args:
            file: filname that exists in bucket
            bucket: bucketname

        Returns:
            Not defined yet

        Example:
            s3.delete_file("learn.txt")

        """
        self.S3.delete_object(Bucket=bucket, Key=file)

    def load_pickle_model(self, file: str, bucket=AWS_S3_BUCKET):
        """

        Args:
            file: filename of pickled object that exists in bucket
            bucket: bucketname

        Returns:
            Object extracted from picked file

        Example:
            myobj = s3.load_pickle_model("Regression.pkl",BUCKET_NAME)
            myobj.predict(x)

        """
        data = self.load_file(file=file, bucket=bucket)
        myfile = open(f"{self.folder}{file}", "wb")
        myfile.write(data)
        with open(f"{self.folder}{file}", "rb") as myfile:
            model = joblib.load(myfile)
        return model

    def store_pickle_model(
        self, model_object: "Model Object", file: str, bucket=AWS_S3_BUCKET
    ):
        """

        Args:
            model_object: Object that needs to be pickled and stored into S3
            file: filename for object
            bucket: bucketname

        Returns:
            Not defined yet

        Example:
            from sklearn.linear_model import LinearRegression
            lm = LinearRegression()
            s3.store_pickle_model(lm,"Regression.pkl")
        """
        joblib.dump(model_object, f"{self.folder}{file}")
        self.store_file(
            filesource=f"{self.folder}{file}", bucket=bucket, filedestination=file
        )

    @staticmethod
    def load_csvdataframe(self, file: str, bucket=AWS_S3_BUCKET):
        data = self.load_file(file=file, bucket=bucket)
        return pd.read_csv(io.BytesIO(data))

    
    def store_csvdataframe(
        self, dataframe: pd.DataFrame, key: str, bucket=AWS_S3_BUCKET
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
        with io.StringIO() as csv_buffer:
            dataframe.to_csv(csv_buffer, index=False)

            response = self.S3.put_object(
                Bucket=bucket, Key=key, Body=csv_buffer.getvalue()
            )

            status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

            if status == 200:
                print(f"Successful S3 put_object response. Status - {status}")
            else:
                print(f"Unsuccessful S3 put_object response. Status - {status}")
