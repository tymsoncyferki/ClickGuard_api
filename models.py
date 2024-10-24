from pydantic import BaseModel

class HTMLPayload(BaseModel):
    url: str
    html: str

class Article(BaseModel):
    title: str
    content: str

class PredictionResponse(BaseModel):
    prediction: int

class DetectionResponse(BaseModel):
    predictions: dict
