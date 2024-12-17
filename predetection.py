from bs4 import BeautifulSoup

from dtos import HTMLPayload, DetectionResponse, ConfName
from utils import get_configuration_name
from prediction_async import predict_titles_async
from config import logger

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
    predictions = await predict_titles_async(titles)
    logger.info(f"google pre-click predictions: {len(predictions.keys())}")
    return DetectionResponse(predictions=predictions)

async def handle_thesun_detection(html_content: str) -> DetectionResponse:
    soup = BeautifulSoup(html_content, 'html.parser')
    titles = {}
    containers = []
    class_names = ['sun-grid-container', 'grids-container', 'content-container']
    for name in class_names:
        containers.extend(soup.find_all('div', {'class': name}))
    for result in containers:
        anchors = result.find_all('a')
        for anchor in anchors:
            if anchor.has_attr('data-headline'):
                title = anchor.get('data-headline')
                link = anchor.get('href')
                if len(title) > 0:
                    titles[link] = title
    # predictions = titles_predict(titles)
    predictions = await predict_titles_async(titles)
    logger.info(f"the sun pre-click predictions: {len(predictions.keys())}")
    return DetectionResponse(predictions=predictions)

async def handle_predetection(payload: HTMLPayload) -> DetectionResponse:
    conf_name = get_configuration_name(payload.url)
    if conf_name == ConfName.GOOGLE:
        return await handle_google_detection(payload.html)
    elif conf_name == ConfName.THESUN:
        return await handle_thesun_detection(payload.html)
    return DetectionResponse(predictions={})