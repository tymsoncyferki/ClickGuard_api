import asyncio
import aiohttp

import numpy as np
from .config import Config
from .measure import calculate_metrics
from .config import MODEL

async def fetch_embedding(session, title):
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {Config.OPEN_API_KEY}"
    }
    payload = {
        "model": "text-embedding-3-large",
        "dimensions": 1000,
        "input": title
    }

    try:
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return data["data"][0]["embedding"]
            else:
                error_message = await response.text()
                raise Exception(f"Error {response.status}: {error_message}")
    except Exception as e:
        print(f"error during fetching embedding for title '{title}': {e}")
        return None


async def get_embeddings(titles):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_embedding(session, title) for title in titles]
        embeddings = await asyncio.gather(*tasks)
        return embeddings
    

async def process_request(titles_dict: dict):
    predictions = {}
    embeddings = await get_embeddings(titles_dict.values())
    for link, title, emb in zip(list(titles_dict.keys()), list(titles_dict.values()), embeddings):
        metrics_dict = calculate_metrics(title)
        for _, value in metrics_dict.items():
            emb.append(value)
        prob = MODEL.predict_proba([emb])[0][1]
        predictions[link] = 1 if prob > 0.5 else 0
    return predictions
