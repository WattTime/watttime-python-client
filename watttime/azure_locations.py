from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import SubscriptionClient
import pandas as pd

def authenticate():
    return DefaultAzureCredential()

def create_client(credential):
    return  SubscriptionClient(credential)

def list_locations(subscription_id, credential=None, subscription_client=None):
    """
    Lists the available Azure locations for a given subscription.
    Args:
        subscription_id (str): The ID of the Azure subscription.
    Returns:
        None
    Prints the available locations with their Azure Region, Display Name, Latitude, and Longitude.
    """

    # Use provided credential or authenticate if not provided
    if credential is None:
        credential = authenticate()

    # Use provided client or create a new client if not provided
    if subscription_client is None:
        subscription_client = create_client(credential)


    # List Locations
    locations = subscription_client.subscriptions.list_locations(subscription_id)

    # Create an empty dataframe
    location_list = []

    location_list = [location for location in locations]
        
    return location_list