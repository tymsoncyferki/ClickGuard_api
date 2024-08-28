from flask import Flask, request, jsonify, render_template
from pydantic import BaseModel, ValidationError
from flask_cors import CORS

from service import handle_predict, handle_extract

app = Flask(__name__)
CORS(app)

class Article(BaseModel):
    url: str
    content: str

@app.route("/", methods=["GET"])
def root():
    return render_template("root.html")

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

@app.route("/extract", methods=["POST"])
def extract():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data = request.get_json()
        article = Article(**data)
        extracted_text = handle_extract(article.content)
        return jsonify({"content": extracted_text})
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
