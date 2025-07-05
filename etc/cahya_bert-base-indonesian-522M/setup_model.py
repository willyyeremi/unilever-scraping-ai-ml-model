import os
import argparse
from transformers import BertTokenizer, BertForSequenceClassification

def main(model_name: str, save_path: str):
    os.makedirs(save_path, exist_ok=True)
    
    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_path)

    model = BertForSequenceClassification.from_pretrained(model_name)
    model.save_pretrained(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to save model files")
    parser.add_argument("--model-name", type=str, default="cahya/bert-base-indonesian-522M")
    args = parser.parse_args()
    main(model_name=args.model_name, save_path=args.model_path)
