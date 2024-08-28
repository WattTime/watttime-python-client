import unittest
from unittest.mock import patch
from watttime.azure_locations import authenticate 

class TestAuthenticate(unittest.TestCase):

    @patch('watttime.azure_locations.DefaultAzureCredential')
    def test_authenticate(self, mock_default_credential):
        # Call the function under test
        credential = authenticate()

        # Verify that DefaultAzureCredential was called
        mock_default_credential.assert_called_once()

        # Verify that the returned value is the mocked credential
        self.assertEqual(credential, mock_default_credential.return_value)

if __name__ == '__main__':
    unittest.main()
