import unittest
from watttime.azure_locations import list_locations 
import os

class TestAzureIntegration(unittest.TestCase):

    def integration_test_list_locations(self):
        subscription_id = os.getenv("SUBS_ID")
        locations = list_locations(subscription_id)

        # Basic check to ensure locations were returned
        self.assertGreater(len(locations), 0)
        self.assertTrue(hasattr(locations[0], 'name'))

if __name__ == '__main__':
    unittest.main()