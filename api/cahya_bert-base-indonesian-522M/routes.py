import os

from flask import Blueprint, jsonify, request
from transformers import BertTokenizer, BertForSequenceClassification
import torch


##############################
# common used variable
##############################

# url = create_url(ordinal = 1, database_product = "postgresql")
# engine = create_engine(url)

predict_bp = Blueprint("data", __name__, url_prefix = "/predict")


##############################
# Load model and tokenizer
##############################

MODEL_PATH = os.environ.get("MODEL_PATH", "./base_model")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  


##############################
# routing function
##############################

@predict_bp.route("/check", methods = ["GET"])
def index():
    return jsonify({"response": "AI Model API (cahya/bert-base-indonesian-522M) is running."}), 200

@predict_bp.route("/process", methods = ["POST"])
def predict():
    # data = request.get_json()
    # text = data.get("text")

    # if not text:
    #     return jsonify({"error": "Missing 'text' in request."}), 400

    # # Tokenisasi dan inference
    # inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # with torch.no_grad():
    #     outputs = model(**inputs)
    #     logits = outputs.logits
    #     predicted_class_id = torch.argmax(logits, dim=-1).item()

    # return jsonify({
    #     "input": text,
    #     "prediction": predicted_class_id
    # })
    return NotImplementedError