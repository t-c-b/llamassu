import torch
import gradio as gr
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
from sys import argv
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline

src=argv[1]
model = AutoModelForCausalLM.from_pretrained(
        src, 
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        revision="main"
)
tokenizer = AutoTokenizer.from_pretrained(src, use_fast=True)
"""
llama_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        #device_map="auto",
        #torch_dtype=torch.float16,
        #trust_remote_code=True,
)
"""
sd_pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        device_map="auto",
        torch_dtype=torch.float16,
        use_safetensors=True,
)
#sd_pipeline.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)

def generate_monster(name):
    """
    monster = llama_pipeline(
            name,
            temperature=0.7,
            do_sample=True, 
            top_p=0.95,
            top_k=40,
            num_return_sequences=1, 
            max_new_tokens=256,
    )[0]['generated_text']
    """
    input_ids = tokenizer(name, return_tensors='pt').input_ids.cuda()
    monster=tokenizer.decode(model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)[0], skip_special_tokens=True)

    #get everything after the table

    torch.cuda.empty_cache()
    return monster

def generate_image(monster):
    prose = monster[monster.rfind("|"):]
    sd_prompt = f"b&w, pen and ink, 80s fantasy, larry elmore, {prose}"
    picture = sd_pipeline(sd_prompt, num_inference_steps=10)
    torch.cuda.empty_cache()
    return picture

with gr.Blocks() as demo:
    inp = gr.Text()
    btn = gr.Button("Generate!")
    with gr.Row():
        out = gr.Markdown()
        img = gr.Image()
    btn.click(fn=generate_monster, inputs=inp, outputs=out)
    out.change(fn=generate_image, inputs=out, outputs=img)

demo.launch()
