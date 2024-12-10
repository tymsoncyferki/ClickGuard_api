# from openai import OpenAI
import pickle
import requests

from .config import Config, MODEL
from .measure import calculate_metrics

def send_request(prompt):
    """ sends request to openai """
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

def return_embeddings_chat(prompt):
    """ gets openai embeddings for text """
    res = send_request(prompt)
    try:
        returned_data = res["data"]
    except ValueError as e:
        returned_data = send_request(prompt)["data"]
    return returned_data[0]["embedding"]


def get_probability_of_clickbait_title(title, metrics_dict=None, model=MODEL):
    """ makes prediction for title using pretrained model """

    # get openai embeddings
    input_to_model = return_embeddings_chat(title)

    # calculate metrics if needed
    if metrics_dict is None:
        metrics_dict = calculate_metrics(title)

    # append metrics to model input
    for _, value in metrics_dict.items():
        input_to_model.append(value)

    return model.predict_proba([input_to_model])[0][1]