import unittest
from unittest.mock import patch, MagicMock
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import SubscriptionClient
from watttime.azure_locations import list_locations
import os

class MockLocation:
    def __init__(self, name, display_name, latitude, longitude):
        self.name = name
        self.display_name = display_name
        self.metadata = type('Metadata', (), {'latitude': latitude, 'longitude': longitude})()

class TestAzureLocations(unittest.TestCase):
    def setUp(self):
        self.subscription_id = os.getenv("SUBS_ID")

    @patch('azure.mgmt.resource.SubscriptionClient')
    @patch('azure.identity.DefaultAzureCredential')
    def test_list_locations(self, mock_default_credential, mock_subscription_client):
        # Mock the DefaultAzureCredential
        mock_credential = MagicMock()
        mock_default_credential.return_value = mock_credential

        # Mock the SubscriptionClient and its methods
        mock_subscription = MagicMock()
        mock_subscription_client.return_value = mock_subscription

        # Mock the list_locations method to return a list of mock locations
        mock_location1 = MockLocation(name='location1', display_name='Location 1', latitude='1.234', longitude='5.678')
        mock_location2 = MockLocation(name='location2', display_name='Location 2', latitude=None, longitude=None)
        mock_locations = [mock_location1, mock_location2]
        mock_subscription.subscriptions.list_locations.return_value = mock_locations 

        # Call the function under test
        locations = list_locations(self.subscription_id) 

        # Assertions to verify the results
        # self.assertEqual(len(locations), 91)
        # self.assertEqual(locations[0].name, 'eastus')
        # self.assertEqual(locations[0].display_name, 'East US')
        # self.assertEqual(locations[0].metadata.latitude, '37.3719')
        # self.assertEqual(locations[0].metadata.longitude, '-79.8164')
        # self.assertEqual(locations[1].name, 'southcentralus')
        # self.assertEqual(locations[1].display_name, 'South Central US')

        # Assertions to verify the structure and content format
        # This approach makes the test resilient to changes in the list of Azure locations, 
        # as long as the data structure remains consistent.
        self.assertIsInstance(locations, list)
        for location in locations:
            self.assertIsInstance(location.name, str)
            self.assertIsInstance(location.display_name, str)
            self.assertTrue(hasattr(location, 'metadata'))
            self.assertTrue(hasattr(location.metadata, 'latitude'))
            self.assertTrue(hasattr(location.metadata, 'longitude'))

            # Check if latitude and longitude are valid formats (e.g., numeric or None)
            if location.metadata.latitude is not None:
                self.assertTrue(isinstance(location.metadata.latitude, str))
                self.assertTrue(location.metadata.longitude.replace('.', '', 1).replace('-', '', 1).isdigit())
            if location.metadata.longitude is not None:
                self.assertTrue(isinstance(location.metadata.longitude, str))
                self.assertTrue(location.metadata.longitude.replace('.', '', 1).replace('-', '', 1).isdigit())


if __name__ == '__main__':
    unittest.main()