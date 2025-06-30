import os
import json
import torch
from utils.tokenize import tokenize_llama_chat, load_model_and_tokenizer, set_seed, tokenize_llama_instruct
from datasets import load_dataset
import argparse
import numpy as np
from utils.generation_probs import *
from tqdm import tqdm
from decoding import monitor_decode
import time
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# GSM8K specific constants and functions
ANSWER_TRIGGER = "The answer is:"

def create_demo_text(n_shot=8, cot_flag=True, shuffle=False):
    question, chain, answer = [], [], []
    question.append("There are 15 trees in the grove. "
                    "Grove workers will plant trees in the grove today. "
                    "After they are done, there will be 21 trees. "
                    "How many trees did the grove workers plant today?")
    chain.append("There are 15 trees originally. "
                 "Then there were 21 trees after some more were planted. "
                 "So there must have been 21 - 15 = 6.")
    answer.append("6")

    question.append(
        "If there are 3 cars in the parking lot and 2 more cars arrive, "
        "how many cars are in the parking lot?")
    chain.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
    answer.append("5")

    question.append(
        "Leah had 32 chocolates and her sister had 42. If they ate 35, "
        "how many pieces do they have left in total?")
    chain.append("Originally, Leah had 32 chocolates. "
                 "Her sister had 42. So in total they had 32 + 42 = 74. "
                 "After eating 35, they had 74 - 35 = 39.")
    answer.append("39")

    question.append(
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
        "has 12 lollipops. How many lollipops did Jason give to Denny?")
    chain.append(
        "Jason started with 20 lollipops. Then he had 12 after giving some "
        "to Denny. So he gave Denny 20 - 12 = 8.")
    answer.append("8")

    question.append(
        "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?")
    chain.append(
        "Shawn started with 5 toys. If he got 2 toys each from his mom and "
        "dad, then that is 4 more toys. 5 + 4 = 9.")
    answer.append("9")

    question.append(
        "There were nine computers in the server room. Five more computers "
        "were installed each day, from monday to thursday. "
        "How many computers are now in the server room?")
    chain.append(
        "There were originally 9 computers. For each of 4 days, 5 more "
        "computers were added. So 5 * 4 = 20 computers were added. "
        "9 + 20 is 29.")
    answer.append("29")

    question.append(
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?")
    chain.append(
        "Michael started with 58 golf balls. After losing 23 on tuesday, "
        "he had 58 - 23 = 35. After losing 2 more, "
        "he had 35 - 2 = 33 golf balls.")
    answer.append("33")

    question.append("Olivia has $23. She bought five bagels for $3 each. "
                    "How much money does she have left?")
    chain.append("Olivia had 23 dollars. "
                 "5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
                 "So she has 23 - 15 dollars left. 23 - 15 is 8.")
    answer.append("8")

    # randomize order of the examples ...
    index_list = list(range(len(question)))
    if shuffle:
        random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list[:n_shot]:
        if cot_flag:
            demo_text += "Q: " + question[i] + "\nA: " + chain[i] + " " + \
                         ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
        else:
            demo_text += "Question: " + question[i] + "\nAnswer: " + \
                         ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
    return demo_text

def build_prompt(input_text, n_shot, cot_flag, shuffle):
    demo = create_demo_text(n_shot, cot_flag, shuffle)
    input_text_prompt = demo + "Q: " + input_text + "\n" + "A:"
    return input_text_prompt

def add_dict_to_json(file_path, new_dict):
    if not os.path.exists(file_path):
        with open(file_path, "w") as file:
            json.dump({}, file, indent=4)

    with open(file_path, "r") as file:
        data = json.load(file)

    data.update(new_dict)

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

def main(args):
    print(args)
    set_seed(43)
    
    # Initialize models based on the approach
    if args.use_dg_decode:
        # Use the monitor_decode approach from generate_text2.py
        llm = monitor_decode(args.base_model)
        model, tokenizer = llm.model_base, llm.tokenizer
        model_expert = llm.model_expert
    else:
        # Use the direct model loading approach from generate_text_gsm8k.py
        if args.base_model == 'Llama-2-7b':
            model, tokenizer = load_model_and_tokenizer(model_name=f"meta-llama/Llama-2-7b-chat-hf")
            tokenizer.pad_token = tokenizer.eos_token
            model_expert, _ = load_model_and_tokenizer(model_name=f"meta-llama/Llama-2-7b-chat-hf")
        elif args.base_model == 'Llama-3-8b':
            model, tokenizer = load_model_and_tokenizer(model_name="meta-llama/Llama-3.1-8B-Instruct")
            tokenizer.pad_token = tokenizer.eos_token
            model_expert, _ = load_model_and_tokenizer(model_name="meta-llama/Llama-3.1-70B-Instruct")
        elif args.base_model == 'gemma-2b':
            model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", device_map="cuda", torch_dtype=torch.bfloat16)
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
            tokenizer.pad_token = tokenizer.eos_token
            model_expert = AutoModelForCausalLM.from_pretrained("google/gemma-2-27b-it", device_map="cuda", torch_dtype=torch.bfloat16)

    # Load dataset based on task
    if args.data_task == 'gsm8k':
        ds = load_dataset("openai/gsm8k", "main")
        data_num = len(ds['test'])
    elif args.data_task == 'truthfulqa':
        task = 'generation'
        ds = load_dataset("truthfulqa/truthful_qa", task)
        data_num = int(len(ds['validation'])/2)
    elif args.data_task == 'triviaqa':
        ds = load_dataset("mandarjoshi/trivia_qa", "rc")
        data_num = 1200
    elif args.data_task == 'nqopen':
        ds = load_dataset("google-research-datasets/nq_open")
        data_num = 1000

    for data_id in tqdm(range(data_num)):
        output = {}
        print(f'# ------------------------------ {data_id} ------------------------------')
        
        # Get data and best answer based on task
        if args.data_task == 'gsm8k':
            data = ds['test']['question'][data_id]
            best_answer = ds['test']['answer'][data_id]
            ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
            best_answer = ANS_RE.search(best_answer).group(1).strip().replace(",", "")
        elif args.data_task == 'truthfulqa':
            data = ds['validation']['question'][data_id]
            best_answer = ds['validation']['best_answer'][data_id]
        elif args.data_task == 'triviaqa':
            data = ds['validation']['question'][data_id]
            best_answer = ds['validation']['answer'][data_id]['aliases']
        elif args.data_task == 'nqopen':
            data = ds['train']['question'][data_id]
            best_answer = ds['train']['answer'][data_id]
        elif args.data_task == 'xsum':
            data = ds['validation']['document'][data_id]
            best_answer = ds['validation']['summary'][data_id]

        # Prepare input tokens based on task and model
        if args.data_task == 'gsm8k':
            input_text = build_prompt(data, 1, 'False', 'False')
            if args.base_model == 'Llama-2-7b':
                input_tokens = tokenize_llama_chat(tokenizer, input_text, None, None)
            elif args.base_model == 'Llama-3-8b':
                input_tokens = tokenize_llama_instruct(tokenizer, input_text, model_output=None, system_prompt=None).to('cuda')
            elif args.base_model == 'gemma-2b':
                input_tokens = tokenizer.encode(input_text, return_tensors='pt').to('cuda')
        else:
            # For other tasks, use the approach from generate_text2.py
            fix_output = "The answer is: "
            system_prompt = 'Please answer the question briefly in 1-2 sentences.'
            if args.data_task == 'xsum':
                system_prompt = 'Please summarize the given content briefly in 1-2 sentences.'
            
            if args.base_model == 'Llama-2-7b':
                input_tokens = tokenize_llama_chat(tokenizer, data, model_output=fix_output, system_prompt=system_prompt).to('cuda')
            elif args.base_model == 'Llama-3-8b':
                input_tokens = tokenize_llama_instruct(tokenizer, data, model_output=fix_output, system_prompt=system_prompt).to('cuda')
            elif args.base_model == 'gemma-2b':
                input_text = system_prompt + ' ' + data + ' ' + fix_output
                input_tokens = tokenizer.encode(input_text, return_tensors='pt').to('cuda')

        fix_input_length = input_tokens.shape[1]

        # Generate base model output
        generated_tokens = model.generate(input_tokens.to(model.device), max_new_tokens=256, top_k=1, do_sample=False, max_length=2096)
        true_generated_tokens = generated_tokens[:, fix_input_length:]
        output_org = tokenizer.batch_decode(true_generated_tokens, skip_special_tokens=True)[0]
        if args.data_task == 'gsm8k':
            output_org = output_org.split('Q')[0]

        # Generate expert model output
        if args.use_dg_decode:
            generated_tokens_expert = model_expert.generate(input_tokens, max_new_tokens=256, do_sample=True, temperature=.9)
        else:
            generated_tokens_expert = model_expert.generate(input_tokens.to(model_expert.device), max_new_tokens=256, top_k=1, do_sample=False, max_length=2096)
        true_generated_tokens_expert = generated_tokens_expert[:, fix_input_length:]
        output_exp = tokenizer.batch_decode(true_generated_tokens_expert, skip_special_tokens=True)[0]
        if args.data_task == 'gsm8k':
            output_exp = output_exp.split('Q')[0]
        
        # Clean up expert generation
        del true_generated_tokens_expert
        del generated_tokens_expert
        del generated_tokens

        output_final = llm.decode_generate(input_tokens=input_tokens, depth=args.depth, width=args.width, r=args.r, branch=args.branch)
        if args.data_task == 'gsm8k':
            output_final = output_final.split('Q')[0]

        # Prepare output dictionary
        output[str(data_id)] = {
            'data': data, 
            'best': best_answer, 
            'expert': output_exp, 
            'base': output_org, 
            'decode': output_final
        }

        # Save results
        filename = "_".join([f'base={args.base_model}', f'exp={args.expert_model}', 
                                f'width={args.width}',f'branch={args.branch}', 
                                f'depth={args.depth}', f'n={args.norm}', f'r={args.r}', f'{data_num}', '.json'])
        save_file_path = os.path.join("results", args.data_task, filename)
        
        add_dict_to_json(save_file_path, output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that processes input and output files.")
    parser.add_argument('--width', type=int, required=False, default=4, help='Search width for generation')
    parser.add_argument('--branch', type=int, required=False, default=2, help='Number of candidates to generate')
    parser.add_argument('--depth', type=int, required=False, default=4, help='Length of generated candidates')
    parser.add_argument('--r', type=float, required=False, default=.5, help='Threshold for decoding')
    parser.add_argument('--data_task', type=str, required=False, default="gsm8k", help='The task to perform')
    parser.add_argument('--base_model', type=str, required=False, default='Llama-2-7b', help='Base model name or path')
    parser.add_argument('--expert_model', type=str, required=False, default='Llama-2-70b', help='Expert model name or path')
    parser.add_argument('--use_monitor_decode', action='store_true', help='Use dg_decode approach instead of manual decoding')

    args = parser.parse_args()
    main(args) 