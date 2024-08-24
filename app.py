from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
import trafilatura
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class Article(BaseModel):
    url: str
    content: str

@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Hello World"})

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'OPTIONS':
        return '', 204  # Return 204 No Content for OPTIONS requests
    try:
        data = request.get_json()
        article = Article(**data)
        prediction = handle_predict(article.content)
        return jsonify({"content": prediction})
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400

def handle_predict(content):
    main_content = trafilatura.extract(content)
    if len(main_content) > 100:
        return 1
    else:
        return 0

if __name__ == "__main__":
    app.run(debug=True)
