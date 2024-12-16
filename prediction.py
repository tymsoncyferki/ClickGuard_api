from typing import Any

import requests

from config import Config, MODEL
from measure import calculate_metrics

def send_request(prompt: str) -> dict:
    """
    sends request to OpenAI embeddings endpoint

    Args:
        prompt (str): input text for generating embeddings

    Returns:
        dict: response from API
    """
    res = requests.post(f"https://api.openai.com/v1/embeddings",
        headers = {
          "Content-Type": "application/json",
          "Authorization": f"Bearer {Config.OPEN_API_KEY}"
        },
        json={
          "model": "text-embedding-3-large",
          "dimensions": 1000,
          "input": f"{prompt}"
        }).json()
    return res

def return_embeddings_chat(prompt: str) -> list:
    """
    gets OpenAI embeddings for a given text prompt

    Args:
        prompt (str): the input text for generating embeddings

    Returns:
        list: embedding vector
    """
    res = send_request(prompt)
    try:
        returned_data = res["data"]
    except ValueError as e:
        returned_data = send_request(prompt)["data"]
    return returned_data[0]["embedding"]


def get_probability_of_clickbait_title(title: str, metrics_dict: dict | None = None, model: Any = MODEL):
    """
    makes prediction for a title using a pretrained model

    Args:
        title (str): title to predict
        metrics_dict (dict, optional): a dictionary of precomputed metrics for the title, if None calculated on the spot
        model (Any): pretrained model used for prediction

    Returns:
        float: probability of the title being clickbait
    """
    # get openai embeddings
    input_to_model = return_embeddings_chat(title)

    # calculate metrics if needed
    if metrics_dict is None:
        metrics_dict = calculate_metrics(title)

    # append metrics to model input
    for _, value in metrics_dict.items():
        input_to_model.append(value)

    return model.predict_proba([input_to_model])[0][1]