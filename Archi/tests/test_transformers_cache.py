import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DynamicCache
import copy

#model_id = "microsoft/Phi-3-mini-128k-instruct" #
#model_id = "EleutherAI/gpt-neox-20b"
#model_id = "mistralai/Mistral-7B-Instruct-v0.2"
model_id = "togethercomputer/LLaMA-2-7B-32K"
#model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})

# Prepare input
'''
prompt = "The capital of France is"
option_a = "Paris"
option_b = "London"
'''
prompt = "What is the capital of France?"
option_a = "I think it is Paris."
option_b = "I think it is London."

# Tokenize inputs
tokenizer.padding_side='left'
#prompt_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
prompt_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
tokenizer.padding_side='right'
option_a_ids = tokenizer.encode(option_a, return_tensors="pt", add_special_tokens=False)
option_b_ids = tokenizer.encode(option_b, return_tensors="pt", add_special_tokens=False)

def get_token_probabilities(logits, token_ids):
    probs = torch.nn.functional.softmax(logits, dim=-1)
    token_probs = [probs[0, i, token_ids[0, i]].item() for i in range(token_ids.size(1))]
    return token_probs

def calculate_sentence_likelihood(token_probs):
    return math.exp(sum(math.log(p) for p in token_probs))

# Function to get model output and logits
def get_output_and_logits(input_ids, cache=None, use_cache=True, position_ids=None,cache_position=None):
    with torch.no_grad():
        outputs = model(input_ids=input_ids, past_key_values=cache, use_cache=use_cache, position_ids=position_ids, cache_position=cache_position)
    return outputs.logits, outputs.past_key_values

# Process prompt and get cache
cache = DynamicCache()
_, cache = get_output_and_logits(prompt_ids, cache=cache, use_cache=True)
#cache = DynamicCache.from_legacy_cache(past_key_values)

# Process options with different position_ids configurations
def process_option(option_ids, cache, prompt_length):
    # Default behavior (no explicit position_ids)
    cache_position = torch.arange(prompt_length,prompt_length+option_ids.shape[1])
    logits_default, _ = get_output_and_logits(option_ids, cache, cache_position=cache_position)
    probs_default = get_token_probabilities(logits_default, option_ids)
    
    # Continuing position_ids from prompt
    position_ids_cont = torch.arange(prompt_length, prompt_length + option_ids.size(1)).unsqueeze(0)
    logits_cont, _ = get_output_and_logits(option_ids, cache, position_ids=position_ids_cont, cache_position=cache_position)
    probs_cont = get_token_probabilities(logits_cont, option_ids)
    
    # Resetting position_ids
    position_ids_reset = torch.arange(option_ids.size(1)).unsqueeze(0)
    logits_reset, _ = get_output_and_logits(option_ids, cache, position_ids=position_ids_reset, cache_position=cache_position)
    probs_reset = get_token_probabilities(logits_reset, option_ids)
    
    return probs_default, probs_cont, probs_reset

'''
probs_a_default, probs_a_cont, probs_a_reset = process_option(option_a_ids, cache, prompt_ids.size(1))
probs_b_default, probs_b_cont, probs_b_reset = process_option(option_b_ids, cache, prompt_ids.size(1))
'''
probs_a_default, probs_a_cont, probs_a_reset = process_option(option_a_ids, copy.deepcopy(cache), prompt_ids.size(1))
probs_b_default, probs_b_cont, probs_b_reset = process_option(option_b_ids, copy.deepcopy(cache), prompt_ids.size(1))

# Process full sequence
full_prompt_a = prompt + " " + option_a
full_prompt_b = prompt + " " + option_b
tokenizer.padding_side='right'
#full_input_a = tokenizer.encode(full_prompt_a, return_tensors="pt")
full_input_a = tokenizer.encode(full_prompt_a, return_tensors="pt", add_special_tokens=False)
#full_input_b = tokenizer.encode(full_prompt_b, return_tensors="pt")
full_input_b = tokenizer.encode(full_prompt_b, return_tensors="pt", add_special_tokens=False)
assert (full_input_a == torch.cat([prompt_ids, option_a_ids], dim=1)).all()
print(full_input_a)
print(prompt_ids)
print(option_a_ids)

def get_option_probs(full_input, option_length):
    with torch.no_grad():
        outputs = model(full_input, use_cache=False)
        logits = outputs.logits
        option_start = full_input.size(1) - option_length
        #option_logits = logits[:, option_start-1:option_start-1+option_length, :]
        option_logits = logits[:, option_start:, :]
        option_ids = full_input[:, option_start:]
        return get_token_probabilities(option_logits, option_ids)

probs_a_full = get_option_probs(full_input_a, option_a_ids.size(1))
probs_b_full = get_option_probs(full_input_b, option_b_ids.size(1))
print(probs_a_full)
print(probs_a_default, probs_a_cont)
# Calculate sentence likelihoods
likelihood_a_default = calculate_sentence_likelihood(probs_a_default)
likelihood_a_cont = calculate_sentence_likelihood(probs_a_cont)
likelihood_a_reset = calculate_sentence_likelihood(probs_a_reset)
likelihood_a_full = calculate_sentence_likelihood(probs_a_full)

likelihood_b_default = calculate_sentence_likelihood(probs_b_default)
likelihood_b_cont = calculate_sentence_likelihood(probs_b_cont)
likelihood_b_reset = calculate_sentence_likelihood(probs_b_reset)
likelihood_b_full = calculate_sentence_likelihood(probs_b_full)

# Print results
print(f"Sentence Likelihoods for Option A ('{option_a}'):")
print(f"Default:   {likelihood_a_default:.6e}")
print(f"Continued: {likelihood_a_cont:.6e}")
print(f"Reset:     {likelihood_a_reset:.6e}")
print(f"Full:      {likelihood_a_full:.6e}")

print(f"\nSentence Likelihoods for Option B ('{option_b}'):")
print(f"Default:   {likelihood_b_default:.6e}")
print(f"Continued: {likelihood_b_cont:.6e}")
print(f"Reset:     {likelihood_b_reset:.6e}")
print(f"Full:      {likelihood_b_full:.6e}")

# Calculate relative differences
def relative_difference(a, b):
    return abs(a - b) / max(abs(a), abs(b))

print("\nRelative differences from full sequence processing:")
print(f"Option A:")
print(f"Default:   {relative_difference(likelihood_a_default, likelihood_a_full):.6f}")
print(f"Continued: {relative_difference(likelihood_a_cont, likelihood_a_full):.6f}")
print(f"Reset:     {relative_difference(likelihood_a_reset, likelihood_a_full):.6f}")

print(f"\nOption B:")
print(f"Default:   {relative_difference(likelihood_b_default, likelihood_b_full):.6f}")
print(f"Continued: {relative_difference(likelihood_b_cont, likelihood_b_full):.6f}")
print(f"Reset:     {relative_difference(likelihood_b_reset, likelihood_b_full):.6f}")

