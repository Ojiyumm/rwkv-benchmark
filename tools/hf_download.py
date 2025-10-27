from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("nvidia/OpenReasoning-Nemotron-7B")
model = AutoModelForCausalLM.from_pretrained("nvidia/OpenReasoning-Nemotron-7B")