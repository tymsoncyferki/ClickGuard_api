from pydantic import BaseModel

class HTMLPayload(BaseModel):
    url: str
    html: str

class Article(BaseModel):
    title: str
    content: str

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    explanation: str
    spoiler: str

class DetectionResponse(BaseModel):
    predictions: dict
