from transformers import AutoTokenizer, LlamaForCausalLM
import torch

# Load the model and tokenizer
model_path = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
model.to('cuda')

# Define a prompt
prompt = "Translate the following text to French: Hello, how are you?"

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

# Generate text with limited resources
with torch.no_grad():
    outputs = model.generate(inputs["input_ids"], max_length=5000, num_return_sequences=1)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the generated text
print(generated_text)


def llama323binstruct_generate_respone(prompt):
    model_path = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model.to('cuda')
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

    # Generate text with limited resources
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=5000, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)
    return generated_text

