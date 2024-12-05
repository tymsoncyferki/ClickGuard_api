import trafilatura
from bs4 import BeautifulSoup
import random
import re

from models import HTMLPayload, Article, PredictionResponse, DetectionResponse, ConfName, CONF_MAPPER

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

def predict(title, content):
    # some ML model
    print(f"Title: {title}")
    print(f"Content: {content}")
    if len(title) * 15 > len(content):
        return random.uniform(0.5, 1)
    else:
        return random.uniform(0, 0.5)

def explain(title, content):
    if len(title) * 15 > len(content):
        return "article content is super short"
    else:
        return "article content is long and detailed"

def spoil(title, content):
    one_word = random.choice(title.split())
    words = content.split()
    k = min(5, len(words))
    chosen_words = random.choices(words, k=k)
    return " ".join([one_word] + chosen_words)

def handle_extract(payload: HTMLPayload):
    html_content = payload.html
    title = extract_title(html_content)
    main_content = extract_content(html_content)
    return Article(title=title, content=main_content)

def handle_predict(payload: Article):
    probability = predict(payload.title, payload.content)
    prediction = 1 if probability > 0.5 else 0
    explanation = explain(payload.title, payload.content)
    spoiler = spoil(payload.title, payload.content)
    return PredictionResponse(prediction=prediction, probability=round(probability, 2), explanation=explanation, spoiler=spoiler)

def handle_extract_and_predict(payload: HTMLPayload):
    article = handle_extract(payload)
    prediction_response = handle_predict(payload=article)
    return prediction_response

def title_predict(title):
    # some ML model
    if len(title) < 35:
        return 1
    else:
        return 0


def matches_pattern(pattern, string):
    escaped_pattern = re.escape(pattern)
    escaped_pattern = escaped_pattern.replace(r'\*', '.*')
    regex_pattern = f'^{escaped_pattern}$'
    return bool(re.match(regex_pattern, string))

def get_configuration_name(url: str) -> ConfName:
    for key, value in CONF_MAPPER.items():
        if matches_pattern(key, url):
            return value
    return ConfName.UNKNOWN

def handle_google_detection(html_content: str) -> DetectionResponse:
    soup = BeautifulSoup(html_content, 'html.parser')
    predictions = {}
    for result in soup.find_all('div', {'class': 'MjjYud'}):
        anchors = result.find_all('a')
        for anchor in anchors:
            title_tag = anchor.find('h3')
            if title_tag is not None:
                title = title_tag.text
                link = anchor.get('href')
                predictions[link] = title_predict(title)
    return DetectionResponse(predictions=predictions)

def handle_thesun_detection(html_content: str) -> DetectionResponse:
    soup = BeautifulSoup(html_content, 'html.parser')
    predictions = {}
    for result in soup.find_all('div', {'class': 'sun-grid-container'}):
        anchors = result.find_all('a')
        for anchor in anchors:
            if anchor.has_attr('data-headline'):
                title = anchor.get('data-headline')
                link = anchor.get('href')
                predictions[link] = title_predict(title)
    return DetectionResponse(predictions=predictions)

def handle_predetection(payload: HTMLPayload) -> DetectionResponse:
    conf_name = get_configuration_name(payload.url)
    if conf_name == ConfName.GOOGLE:
        return handle_google_detection(payload.html)
    elif conf_name == ConfName.THESUN:
        return handle_thesun_detection(payload.html)
    return DetectionResponse(predictions={})