from typing import List
import sys
import os
from transformers import PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer
import torch

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
BASE_INPUT = "Input:"
BASE_RESPONSE = "\nResponse:"

ADD_FROM_POS_CHAT = E_INST
ADD_FROM_POS_BASE = BASE_RESPONSE


def tokenize_llama_chat(
    tokenizer: PreTrainedTokenizer,
    user_input: str,
    model_output: str = None,
    system_prompt: str = None,
) -> List[int]:
    input_content = ""
    if system_prompt is not None:
        input_content += B_SYS + system_prompt + E_SYS
    input_content += f"{B_INST} {user_input.strip()} {E_INST}"
    if model_output is not None:
        input_content += f" {model_output.strip()}"
    return tokenizer.encode(input_content, return_tensors='pt')


def tokenize_llama_base(
    tokenizer, user_input: str, model_output: str = None
) -> List[int]:
    input_content = ""
    input_content += f"{BASE_INPUT} {user_input.strip()}"
    if model_output is not None:
        input_content += f"{BASE_RESPONSE} {model_output.strip()}"
    return tokenizer.encode(input_content)

def tokenize_llama_instruct(tokenizer, user_input, model_output, system_prompt):
    chat = [{"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}]
    input_content = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    if model_output != None:
        input_content += model_output
    return tokenizer.encode(input_content, return_tensors='pt')


# Function to load a model and its tokenizer from a directory or download it if not present
def load_model_and_tokenizer(model_name, device='cuda'):
    """
    Loads or downloads the specified model and its tokenizer.

    Parameters:
    - model_name: str - the name of the model to load
    - device: str - the device type to use ('cuda' or 'cpu')

    Returns:
    - model: PreTrainedModel - the loaded model
    - tokenizer: PreTrainedTokenizer - the loaded tokenizer
    """
    # hf_token = "your access token"  # Replace with your actual token
    hf_token = None
    # cache_dir = "models"  # Directory where models are stored
    cache_dir = '/data/yuruic/hug/'
    model_dir = os.path.join(cache_dir, model_name)  # Path to the specific model directory

    # Define model-specific options
    model_opts = {
        "attn_implementation": "flash_attention_2",
        "torch_dtype": torch.bfloat16
    }

    # If the model is not a '7b' model, add the option to load it in 4-bit precision
    if "7b" not in model_name:
        model_opts["load_in_4bit"] = True

    # Load or download the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name if not os.path.exists(model_dir) else model_dir,
        token=hf_token if not os.path.exists(model_dir) else None,
        cache_dir=cache_dir if not os.path.exists(model_dir) else None,
        device_map='auto',
        output_hidden_states=True,
        **model_opts
    )

    # Load or download the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name if not os.path.exists(model_dir) else model_dir,
        token=hf_token if not os.path.exists(model_dir) else None,
        cache_dir=cache_dir if not os.path.exists(model_dir) else None
    )

    return model, tokenizer




def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False