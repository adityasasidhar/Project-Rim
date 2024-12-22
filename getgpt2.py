from transformers import GPT2Tokenizer, GPT2LMHeadModel


def download_gpt2_model(model_name):
    # Load the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    print(f"{model_name} model downloaded successfully.")
    return tokenizer, model


# Downloading GPT-2 Small (124M parameters)
gpt2_small_tokenizer, gpt2_small_model = download_gpt2_model("gpt2")

# Downloading GPT-2 Medium (355M parameters)
gpt2_medium_tokenizer, gpt2_medium_model = download_gpt2_model("gpt2-medium")

# Downloading GPT-2 Large (774M parameters)
gpt2_large_tokenizer, gpt2_large_model = download_gpt2_model("gpt2-large")

