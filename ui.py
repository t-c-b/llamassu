import gradio as gr
from sys import argv
from diffusers import StableDiffusionPipeline

llama_pipeline=pipeline(model=argv[1])
sd_pipeline=StableDiffusionPipeline(
        "runwayml/stable-diffusion-1-5",
        torch_dtype=torch.float16
        use_safetensors=True
)

def generate_monster(name):
    monster = llama_pipeline(name)

    #get everything after the table
    prose = monster[monster.rfind("|"):]

    sd_prompt = f"{name}, black and white, 80s fantasy, pen and ink, {prose}"
    picture = sd_pipeline(sd_prompt)

    return monster, picture

demo = gr.Interface(fn=generate_monster, inputs="text", outputs=["text","image"])
