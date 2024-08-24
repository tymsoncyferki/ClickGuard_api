from fastapi import FastAPI
from pydantic import BaseModel
import trafilatura

class Article(BaseModel):
    url: str
    content: str

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(article: Article):
    main_content = trafilatura.extract(article.content)
    return {"content": main_content}