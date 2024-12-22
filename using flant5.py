from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the model and tokenizer
model_name = "google/flan-t5-base"  # Replace with "flan-t5-base" for a larger model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def chat_with_flan_t5(user_input):
    # Tokenize the input
    inputs = tokenizer(user_input, return_tensors="pt", max_length=512, truncation=True)

    # Generate response
    outputs = model.generate(
        inputs["input_ids"],
        max_length=150,  # Maximum length of the response
        num_beams=5,  # Beam search for better response quality
        early_stopping=True,
        no_repeat_ngram_size=2
    )

    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# Chatbot interaction loop
print("Chatbot: Hello! How can I assist you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() in {"exit", "quit"}:
        print("Chatbot: Goodbye!")
        break
    response = chat_with_flan_t5(user_input)
    print(f"Chatbot: {response}")
