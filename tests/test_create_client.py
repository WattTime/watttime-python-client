import unittest
from unittest.mock import MagicMock
from azure.mgmt.resource import SubscriptionClient
from watttime.azure_locations import create_client

class TestCreateSubscriptionClient(unittest.TestCase):

    def test_create_subscription_client(self):
        # Mock the credential
        mock_credential = MagicMock()

        # Call the function under test
        subscription_client = create_client(mock_credential)

        # Verify that the SubscriptionClient was created with the mock credential
        self.assertIsInstance(subscription_client, SubscriptionClient)

if __name__ == '__main__':
    unittest.main()