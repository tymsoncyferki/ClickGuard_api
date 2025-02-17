import unittest
from flask import Flask
from ..app import app
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
            "html": "<html><body><h1>Test Article</h1><p>This is a test.</p></body></html>",
            "generateSpoiler": False
        }
        response = self.app.post('/extract_and_predict', json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"prediction", response.data)
        self.assertIn(json.loads(response.data)['prediction'], [0,1]) 
    
    def test_google_detect(self):
        # Test the /predetect endpoint on google example
        with open("test_files/test_google.html", 'rb') as file:
            html = file.read()
        payload = {
            "url": "https://www.google.com/search?q=queryselector+google+search+results&sca_esv=381a7a3a6a330f3",
            "html": str(html)
        }
        response = self.app.post('/predetect', json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"predictions", response.data)
        # print(json.loads(response.data))
        

if __name__ == "__main__":
    unittest.main()
