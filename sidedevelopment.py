import os
import torch
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer
import spacy

# Load SpaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Flask App
app = Flask(__name__)

gpt2tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2model = GPT2LMHeadModel.from_pretrained('gpt2')


def gpt2finetuned_generate_response(prompt):
    inputs = gpt2tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = gpt2model.generate(
            inputs.input_ids,
            attention_mask=inputs['attention_mask'],
            max_length=500,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=gpt2tokenizer.eos_token_id,
        )
    return gpt2tokenizer.decode(outputs[0], skip_special_tokens=True)


# Detect intent (can be extended)
def detect_intent(prompt):
    doc = nlp(prompt.lower())
    if any(token.lemma_ in ["image", "visualize", "art", "picture"] for token in doc):
        return "image_generation"
    return "text_generation"


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    prompt = data.get('prompt', '').strip()
    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400

    intent = detect_intent(prompt)
    if intent == "text_generation":
        response = gpt2finetuned_generate_response(prompt)
        return jsonify({"response": response})
    else:
        # Future enhancement: Handle image generation here
        return jsonify({"error": "Image generation not yet supported"}), 501


if __name__ == '__main__':
    app.run(debug=True)