import unittest
from flask import Flask
from app import app
import json

class TestEndpoints(unittest.TestCase):
    def setUp(self):
        # Set up flask test client
        self.app = app.test_client()
        self.app.testing = True

    def test_extract(self):
        # Test the /extract endpoint
        payload = {
            "url": "http://example.com",
            "html": "<html><body><h1>Test Article</h1><p>This is a test.</p></body></html>"
        }
        response = self.app.post('/extract', json=payload)
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertEqual(response_data['title'], 'Test Article')
        self.assertGreater(len(response_data['content']), len('This is a test'))

    def test_predict(self):
        # Test the /predict endpoint
        payload = {
            "title": "Test Title",
            "content": "This is some test content."
        }
        response = self.app.post('/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"prediction", response.data)

    def test_extract_and_predict(self):
        # Test the /extract_and_predict endpoint
        payload = {
            "url": "http://example.com",
            "html": "<html><body><h1>Test Article</h1><p>This is a test.</p></body></html>"
        }
        response = self.app.post('/extract_and_predict', json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"prediction", response.data)
        self.assertEqual(json.loads(response.data)['prediction'], 1) # with current model

if __name__ == "__main__":
    unittest.main()
