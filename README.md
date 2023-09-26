# Quantized Trainer For RPG Statblocks
Finetune a quantized language model to generate statistics and descriptions for [Monsters from the Basic Fantasy Roleplaying Game](https://basicfantasy.org/srd/monstersAll.html). 

## Convert the statblocks into Markdown
Run `wget https://basicfantasy.org/srd/monstersAll.html` then `htmlreader.py`

## Finetune a LLM locally on a single GPU
Run `finetune_llama_gptq.py`. Modified from the Huggingface example as follows:
- Resume from a checkpoint with `--checkpoint=results/checkpoint-[number]`
- Load text documents from a local directory as the dataset

## Inferencing
Included a simple script to inference interactively `llama.py results/checkpoint-[number]`
