from bs4 import BeautifulSoup

from .dtos import HTMLPayload, DetectionResponse, ConfName
from .utils import get_configuration_name

def title_predict(title):
    # some ML model
    if len(title) < 35:
        return 1
    else:
        return 0

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