from transformers import AutoTokenizer, AutoModelForCausalLM

llama321Binstruct_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
llama321Binstruct_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

llama323Binstruct_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
llama323Binstruct_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")


