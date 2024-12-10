import unittest
import time

from ..predetection import handle_google_detection

class TestPerformance(unittest.IsolatedAsyncioTestCase):

    async def test_google_detect(self):
        with open("test_files/trump_google.html", 'rb') as file:
            html = file.read()
        start_time = time.time()
        response = await handle_google_detection(str(html))
        end_time = time.time()
        self.assertEqual(len(response.predictions.keys()), 9)
        print(f"Execution Time: {end_time - start_time} seconds")

if __name__ == "__main__":
    unittest.main()
