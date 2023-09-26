import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sys import argv

src = argv[1]

model = AutoModelForCausalLM.from_pretrained(
        src, 
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        revision="main"
)
tokenizer = AutoTokenizer.from_pretrained(src, use_fast=True)

while True:
    prompt = input(">")
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
    print(tokenizer.decode(output[0]))
