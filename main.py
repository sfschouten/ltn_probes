from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)


tokenizer = AutoTokenizer.from_pretrained("TheBloke/koala-7B-GPTQ-4bit-128g", use_fast=True)
examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    )
]

quantized_model_dir = '../koala'


# load the quantized model
model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, model_basename="koala-7B-4bit-128g", use_safetensors=True, device="cuda:0")

# or you can also use pipeline
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
print(pipeline("auto-gptq is")[0]["generated_text"])
