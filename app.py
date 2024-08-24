from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
import trafilatura
from flask import Flask, request, jsonify

app = Flask(__name__)

class Article(BaseModel):
    url: str
    content: str

@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Hello World"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        article = Article(**data)
        main_content = trafilatura.extract(article.content)
        return jsonify({"content": main_content})
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
