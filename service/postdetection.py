import trafilatura
from bs4 import BeautifulSoup
import random

from .dtos import HTMLPayload, Article, PredictionResponse
from .measure import explain_baitness, calculate_metrics
from .model import get_probability_of_clickbait_title

# extraction 

def extract_content(html_content):
    main_content = trafilatura.extract(html_content)
    if main_content is None:
        main_content = ""
    return main_content

def extract_title(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    h1_tag = soup.find('h1')
    title = ''
    if h1_tag:
        title = h1_tag.get_text()
    return title

def predict(title, content, metrics_dict=None):
    # some ML model
    # if len(title) * 15 > len(content):
    #     return random.uniform(0.5, 1)
    # else:
    #     return random.uniform(0, 0.5)
    return get_probability_of_clickbait_title(title, metrics_dict=metrics_dict)

def explain(title, content, probability, metrics_dict=None):
    return explain_baitness(title, probability, metrics_dict=metrics_dict)

def spoil(title, content):
    one_word = random.choice(title.split())
    words = content.split()
    k = min(5, len(words))
    chosen_words = random.choices(words, k=k)
    return " ".join([one_word] + chosen_words)

# functions to be imported into api

def handle_extract(payload: HTMLPayload):
    html_content = payload.html
    title = extract_title(html_content)
    main_content = extract_content(html_content)
    return Article(title=title, content=main_content)

def handle_predict(payload: Article):
    metrics_dict = calculate_metrics(payload.title)
    probability = predict(payload.title, payload.content, metrics_dict=metrics_dict)
    prediction = 1 if probability > 0.5 else 0
    explanation = explain(payload.title, payload.content, probability, metrics_dict=metrics_dict)
    spoiler = spoil(payload.title, payload.content)
    return PredictionResponse(prediction=prediction, probability=round(probability, 2), explanation=explanation, spoiler=spoiler)

def handle_extract_and_predict(payload: HTMLPayload):
    article = handle_extract(payload)
    prediction_response = handle_predict(payload=article)
    return prediction_response