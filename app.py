import os
from flask import Flask, request, jsonify
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Inisialisasi Flask
app = Flask(__name__)

# Muat model dan tokenizer
model_dir = "fine_tuned_arabic_question_model"  # Ganti dengan direktori model Anda
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
model.eval()  # Set model ke mode evaluasi

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    context = data.get('context', '')

    # Tokenisasi input
    inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=512).input_ids
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=50)

    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'question': question})

if __name__ == '__main__':
    app.run(port=5000)  # Tentukan port sesuai kebutuhan
