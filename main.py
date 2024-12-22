from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer
from diffusers import StableDiffusionPipeline
import spacy
import os
import torch

nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

# Open the file in read mode ('r') and read its content

def clear_cuda_cache():
    torch.cuda.empty_cache()  # Frees the GPU memory cache
    print("CUDA cache cleared.")


def report_cuda_memory():
    print("CUDA Memory Summary:")
    print(torch.cuda.memory_summary(device=None, abbreviated=False))  # Summarizes GPU memory usage

report_cuda_memory()
clear_cuda_cache()
llama321Binstruct_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", pad_token="[PAD]")
llama321Binstruct_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
report_cuda_memory()

def llama321Binstruct(prompt,context_file=None):
    inputs = llama321Binstruct_tokenizer(prompt, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = llama321Binstruct_model.generate(
            inputs.input_ids,
            attention_mask=inputs['attention_mask'],
            max_length=500,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=llama321Binstruct_tokenizer.eos_token_id,
        )
    return llama321Binstruct_tokenizer.decode(outputs[0], skip_special_tokens=True)

def llama323binstruct(prompt,context_file=None):
    clear_cuda_cache()
    llama323Binstructtokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", pad_token="[PAD]")
    llama323Binstructmodel = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct",
                                                                  torch_dtype=torch.float16)
    llama323Binstructmodel.to('cuda')
    inputs = llama323Binstructtokenizer(prompt, return_tensors="pt", padding=True).to('cuda')
    with torch.no_grad():
        outputs = llama323Binstructmodel.generate(
            inputs["input_ids"],
            attention_mask=inputs['attention_mask'],
            max_length=500,
            num_return_sequences=1,
            pad_token_id=llama323Binstructtokenizer.eos_token_id,
        )
    return llama323Binstructtokenizer.decode(outputs[0], skip_special_tokens=True)

def gpt2finetuned(prompt):
    clear_cuda_cache()
    gpt2tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2model = GPT2LMHeadModel.from_pretrained('gpt2')
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


def stablediffusion_generate_image(prompt,guidance_scale=10, num_inference_steps=100,
                                                seed=None, height=512, width=512):
    clear_cuda_cache()
    model = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4')
    model.to('cuda')
    report_cuda_memory()
    print(f"Generating image...for the prompt: {prompt}")
    image = model(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    ).images[0]
    clear_cuda_cache()
    image.save(f"AI generate images/{prompt}.png")
    return image

def intent(prompt):
    # Predefined keyword sets
    image_keywords = {
        "image", "visualize", "art", "picture", "draw", "illustration", "design",
        "photo", "graphic", "sketch", "render", "painting", "sculpture", "portrait",
        "scene", "visual", "animation", "diagram"
    }

    explanation_keywords = {
        "explain", "detail", "complex", "understand", "why", "how", "clarify",
        "reason", "cause", "elaborate", "define", "meaning", "interpret", "describe",
        "analyze", "overview", "insight", "rationale", "context"
    }

    summarization_keywords = {
        "summarize", "condense", "brief", "short", "abstract", "outline",
        "simplify", "compress", "gist", "recap", "key points", "highlights",
        "summary", "concise", "main ideas", "essence"
    }

    query_keywords = {
        "what", "when", "where", "who", "is", "are", "does", "query",
        "information", "find", "search", "locate", "retrieve", "discover",
        "lookup", "tell me", "show me", "identify", "facts", "details",
        "specific", "guide", "help", "how many", "which"
    }
    doc = nlp(prompt.lower())
    tokens = {token.lemma_ for token in doc}

    # Detect intent
    if tokens & image_keywords:
        return "image_generation"
    elif tokens & explanation_keywords:
        return "explanation"
    elif tokens & summarization_keywords:
        return "summarization"
    elif any(token.lemma_ in query_keywords for token in doc) or doc[-1].lemma_ == "?":
        return "query"
    else:
        return "simple_chat"

def generate_response(prompt):
    if intent(prompt) == "image_generation":
        return stablediffusion_generate_image(prompt)
    elif intent(prompt) == "explanation":
        return llama323binstruct(prompt)
    elif intent(prompt) == "summarization":
        return llama323binstruct(prompt)
    elif intent(prompt) == "query":
        return llama321Binstruct(prompt)
    else:
        return gpt2finetuned(prompt)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/api/chat', methods=['POST'])
def chat():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json."}), 400

    data = request.get_json()
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({"error": "The 'prompt' field is required."}), 400

    try:
        response = generate_response(prompt)
    except Exception as e:
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500

    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(debug=True)
