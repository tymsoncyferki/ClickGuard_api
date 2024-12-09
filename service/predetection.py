from bs4 import BeautifulSoup

from .dtos import HTMLPayload, DetectionResponse, ConfName
from .utils import get_configuration_name
from .measure import get_baitness_scaled
from .async_model import process_request

# def titles_predict(titles):
#     def get_prediction(probability):
#         return 1 if probability > 0.5 else 0
#     return {link: get_prediction(get_baitness_scaled(title)) for link, title in titles.items()}

async def handle_google_detection(html_content: str) -> DetectionResponse:
    soup = BeautifulSoup(html_content, 'html.parser')
    titles = {}
    for result in soup.find_all('div', {'class': 'MjjYud'}):
        anchors = result.find_all('a')
        for anchor in anchors:
            title_tag = anchor.find('h3')
            if title_tag is not None:
                title = title_tag.text
                link = anchor.get('href')
                titles[link] = title
                # predictions[link] = title_predict(title)
    # predictions = titles_predict(titles)
    predictions = await process_request(titles)
    return DetectionResponse(predictions=predictions)

async def handle_thesun_detection(html_content: str) -> DetectionResponse:
    soup = BeautifulSoup(html_content, 'html.parser')
    titles = {}
    for result in soup.find_all('div', {'class': 'sun-grid-container'}):
        anchors = result.find_all('a')
        for anchor in anchors:
            if anchor.has_attr('data-headline'):
                title = anchor.get('data-headline')
                link = anchor.get('href')
                titles[link] = title
    # predictions = titles_predict(titles)
    predictions = await process_request(titles)
    return DetectionResponse(predictions=predictions)

async def handle_predetection(payload: HTMLPayload) -> DetectionResponse:
    conf_name = get_configuration_name(payload.url)
    if conf_name == ConfName.GOOGLE:
        return await handle_google_detection(payload.html)
    elif conf_name == ConfName.THESUN:
        return await handle_thesun_detection(payload.html)
    return DetectionResponse(predictions={})