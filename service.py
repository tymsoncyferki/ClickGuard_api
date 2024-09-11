import trafilatura
from bs4 import BeautifulSoup

from models import HTMLPayload, Article, PredictionResponse

def extract_content(html_content):
    main_content = trafilatura.extract(html_content)
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
        return 1
    else:
        return 0

def handle_extract(payload: HTMLPayload):
    html_content = payload.html
    title = extract_title(html_content)
    main_content = extract_content(html_content)
    return Article(title=title, content=main_content)

def handle_predict(payload: Article):
    prediction = predict(payload.title, payload.content)
    return PredictionResponse(prediction=prediction)

def handle_extract_and_predict(payload: HTMLPayload):
    article = handle_extract(payload)
    prediction_response = handle_predict(payload=article)
    return prediction_response

