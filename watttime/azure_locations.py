from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import SubscriptionClient
import pandas as pd

def authenticate():
    return DefaultAzureCredential()

def create_client(credential):
    return  SubscriptionClient(credential)

def list_locations(subscription_id):
    """
    Lists the available Azure locations for a given subscription.
    Args:
        subscription_id (str): The ID of the Azure subscription.
    Returns:
        None
    Prints the available locations with their Azure Region, Display Name, Latitude, and Longitude.
    """
    # Authenticate
    credential = authenticate()

    # Create a Subscription Client
    subscription_client = create_client(credential)

    # List Locations
    locations = subscription_client.subscriptions.list_locations(subscription_id)

    # Create an empty dataframe
    location_list = []

    location_list = [location for location in locations]
        
    return location_list