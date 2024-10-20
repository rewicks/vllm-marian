from vllm import LLM, SamplingParams
from vllm.inputs import zip_enc_dec_prompts, TokensPrompt

src = [
    "I have to take the dog for a walk.",
]
decoder_prompt = [TokensPrompt(prompt_token_ids=[1, 702, 1075,])]

prompts = zip_enc_dec_prompts(src, decoder_prompt)

sampling_params = SamplingParams(temperature=0, top_p=1.0)

# enforce eager is for debugging and prevents cuda graphs from being built (slower)
llm = LLM(model="rewicks/baseline_en-de_8k_ep1", enforce_eager=True)
# llm = LLM(model="/home/hltcoe/rwicks/code/vllm-marian/MARIAN-README/conversion-test", enforce_eager=True)

outputs = llm.generate(prompts, sampling_params)
 
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print(output.outputs[0].token_ids)


# To get the *correct* output
sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=1)
src = [
    "I have to take the dog for a walk."
]
decoder_prompt_tokens = [1]
decoder_prompt = [TokensPrompt(prompt_token_ids=decoder_prompt_tokens)]
last_token=decoder_prompt_tokens[-1]
text = ""
while last_token != 2:
    prompts = zip_enc_dec_prompts(src, decoder_prompt)
    outputs = llm.generate(prompts, sampling_params)
    decoder_prompt_tokens.append(outputs[0].outputs[0].token_ids[0])
    last_token=decoder_prompt_tokens[-1]
    decoder_prompt = [TokensPrompt(prompt_token_ids=decoder_prompt_tokens)]
    text += outputs[0].outputs[0].text

print(decoder_prompt_tokens)
print(text)