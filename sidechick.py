from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer
from diffusers import StableDiffusionPipeline
import spacy
import torch
import logging

app = Flask(__name__)

def clear_cuda_cache():
    torch.cuda.empty_cache()
    print("CUDA cache cleared.")


def report_cuda_memory():
    print("CUDA Memory Summary:")
    print(torch.cuda.memory_summary(device=None, abbreviated=False))  # Summarizes GPU memory usage


llama321Binstructtokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token=True)
llama321Binstructmodel = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token=True)
llama321Binstructmodel = llama321Binstructmodel.to('cuda')
llama321Binstructtokenizer.pad_token = llama321Binstructtokenizer.eos_token


def llama321binstruct(prompt):
    print("llama has been called")
    # Prepare the input data and ensure it is moved to the same device
    inputs = llama321Binstructtokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to('cuda')
    with torch.no_grad():
        outputs = llama321Binstructmodel.generate(
            inputs["input_ids"],
            attention_mask=inputs['attention_mask'],
            max_length=500,
            num_return_sequences=1,
            pad_token_id=llama321Binstructtokenizer.pad_token_id,
        )
        print("llama response has been generated")
        print(outputs)
    return llama321Binstructtokenizer.decode(outputs[0], skip_special_tokens=True)


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


def generate_response(prompt):
    return llama321binstruct(prompt)

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
