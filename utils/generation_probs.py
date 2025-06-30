import torch
import re
import math
from rouge_score import rouge_scorer
import numpy as np
import os
import json


def product_of_probs_norm(tsr):
    with torch.no_grad():
        return torch.exp(torch.sum(torch.log(tsr), dim=-1)/(tsr.shape[1]+1e-10))


def product_of_probs(tsr):
    with torch.no_grad():
        return torch.exp(torch.sum(torch.log(tsr), dim=-1))


def calculate_entropy(probs):
    # Ensure probabilities are greater than zero to avoid log(0)
    probs = probs.clamp(min=1e-10)
    entropy = -torch.sum(probs * torch.log(probs))
    return entropy


def topk_token_probs(model, true_generated_tokens_temp, input_length):
    output_logits_all = model(true_generated_tokens_temp).logits[:, (input_length-1):-1, :]
    output_logits_all = torch.softmax(output_logits_all, dim=-1)
    t = []
    v = []
    for i in range(output_logits_all.shape[1]):
        v_temp, t_temp = torch.topk(output_logits_all[:, i, :], k=2)
        t.append(t_temp[0, 1].item())
        print(v_temp[0, 1].item())
        v.append(v_temp[0, 1].item())
    
    return torch.tensor(v), t


def generation_probs(model, generated_tokens, input_length):

    true_generated_tokens = generated_tokens[:, input_length:].unsqueeze(-1)
    output_logits_all = model(generated_tokens).logits[:, (input_length-1):-1, :]
    output_probs_all = torch.softmax(output_logits_all, dim=-1)
    output_probs = torch.gather(output_probs_all, 2, true_generated_tokens).squeeze(-1)

    return output_probs


def most_words_in_another_sentence(sentence1, sentence2):
    words_sentence1 = set(re.findall(r"\w+", sentence1.replace("!", ".")))
    words_sentence2 = set(re.findall(r"\w+", sentence2.replace("!", ".")))
    common_words = words_sentence1.intersection(words_sentence2)
    proportion = len(common_words) / min(len(words_sentence1), len(words_sentence2))
    
    return proportion > 0.5


def generate_k_path(model, tokenizer, input_tokens, k):

    input_length = input_tokens.shape[1]
    drop_id = 13 # \n\n
    stop_id = tokenizer.convert_tokens_to_ids('.')
    stop_id1 = tokenizer.convert_tokens_to_ids(').')
    stop_id2 = tokenizer.convert_tokens_to_ids('."')

    possible_generated_output = []
    possible_generated_probs = []
    possible_generated_sen_first = []
    possible_generated_tokens_first = []
    topk_tokens = torch.topk(model(input_tokens).logits[:, -1, :], k=k).indices
    
    for i in range(k):

        input_tokens_temp = torch.concat((input_tokens, topk_tokens[:,i].unsqueeze(0)), dim=-1)
        generated_tokens_temp = model.generate(input_tokens_temp, top_k=1, max_new_tokens=100)
        true_generated_tokens_temp = generated_tokens_temp[:, input_length:]
        possible_generated_output.append(true_generated_tokens_temp)

        # generation probability
        output_probs_temp = generation_probs(model, generated_tokens_temp, input_length)
        possible_generated_probs.append(output_probs_temp)

        paragraph = true_generated_tokens_temp.clone()
        sentences_boundaries = torch.where((paragraph==stop_id) | (paragraph==stop_id1) | (paragraph==stop_id2))[1]
        sentences_boundaries = torch.cat((torch.tensor([-1]).to('cuda'), sentences_boundaries, torch.tensor([paragraph.shape[1]]).to('cuda')), dim=-1)
        
        sentences = []

        for id_s in range(1, sentences_boundaries.shape[0]-1):
            paragraph_part = paragraph[:, sentences_boundaries[id_s-1]+1:(sentences_boundaries[id_s]+1)]
            sentences.append(torch.cat((input_tokens, paragraph_part[:, paragraph_part[0, :] != drop_id]), dim=1))
        
        possible_generated_tokens_first.append(paragraph[:, :sentences_boundaries[1]+1])
        if len(sentences) != 0:
            possible_generated_sen_first.append(tokenizer.batch_decode(sentences[0][:, input_length:])[0].lower())
        else:
            possible_generated_sen_first.append(tokenizer.eos_token)

    return topk_tokens, possible_generated_tokens_first, possible_generated_sen_first, possible_generated_output

            
def most_words_in_another_sentence(sentence1, sentence2):
    words_sentence1 = set(sentence1.split())
    words_sentence2 = set(sentence2.split())
    common_words = words_sentence1.intersection(words_sentence2)
    proportion = len(common_words) / len(words_sentence1)
    
    return proportion > 0.5



def generate_k_path_fix_length(model, tokenizer, input_tokens, k, max_new_tokens):

    # tokenizer.eos_token
    input_length = input_tokens.shape[1]

    all_generated_output = []
    all_generated_probs = []
    all_generated_sen = []
    topk_tokens = torch.topk(model(input_tokens).logits[:, -1, :], k=k).indices
    
    for i in range(k):

        input_tokens_temp = torch.concat((input_tokens, topk_tokens[:,i].unsqueeze(0)), dim=-1)
        generated_tokens_temp = model.generate(input_tokens_temp, top_k=1, max_new_tokens=max_new_tokens)
        all_generated_output.append(generated_tokens_temp)
        
        output_probs_temp = generation_probs(model, generated_tokens_temp, input_length)
        all_generated_probs.append(output_probs_temp)

        true_generated_tokens_temp = generated_tokens_temp[:, input_length:]
        all_generated_sen.append(tokenizer.batch_decode(true_generated_tokens_temp, skip_special_tokens=True)[0])

    return topk_tokens, all_generated_output, all_generated_probs, all_generated_sen


def group_sentences(all_sentences, threshold=0.8):
    sentences = [sen.split('.')[0] for sen in all_sentences]
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    groups = []
    sentence_ids = []
    used_indices = set()

    for i, sentence1 in enumerate(sentences):
        if i in used_indices:
            continue
        group = [sentence1]
        sentence_id = [i]
        used_indices.add(i)
        for j, sentence2 in enumerate(sentences[i+1:], start=i+1):
            if j in used_indices:
                continue
            rouge_score = scorer.score(sentence1, sentence2)['rougeL'].fmeasure
            if rouge_score > threshold:
                group.append(sentence2)
                sentence_id.append(j)
                used_indices.add(j)
        groups.append(group)
        sentence_ids.append(sentence_id)

    return groups, sentence_ids


def speculative_probs(base_tokens, base_probs, model_expert, input_length, normalization):
 
    # model_expert probability
    expert_probs = generation_probs(model_expert, base_tokens, input_length)

    if base_probs.shape[0] == 1:
        base_prob = base_probs[0]
        expert_prob = expert_probs[0]
    else:
        if normalization == 'True':
            base_prob = product_of_probs_norm(base_probs)
            expert_prob = product_of_probs_norm(expert_probs)
        else:
            base_prob = product_of_probs(base_probs)
            expert_prob = product_of_probs(expert_probs)

    final_prob = min(1, expert_prob/base_prob)

    return final_prob, base_prob, expert_prob


def tree_search(model, model_expert, tokenizer, candidates, search_width, num_candidates, candidate_length):
    candidate_probs = []
    candidate_generated_output = []
    for j in range(len(candidates)):
        input_tokens = candidates[j]
        _, all_generated_output, all_generated_probs, _ = generate_k_path_fix_length(model, tokenizer, input_tokens, num_candidates, candidate_length)
        input_length0 =input_tokens.shape[0]
        
        for i in range(num_candidates):
            base_tokens = all_generated_output[i]
            base_probs = all_generated_probs[i]
            final_prob, _, _ = speculative_probs(base_tokens, base_probs, model_expert, input_length0, True)
            candidate_probs.append(final_prob)

        candidate_generated_output.extend(all_generated_output)
    
    width_indices = np.argsort(np.array(candidate_probs))[::-1][:search_width]
    kept_generated_output = [candidate_generated_output[j] for j in width_indices]

    return kept_generated_output, candidate_probs


def add_dict_to_json(file_path, new_dict):

    if not os.path.exists(file_path):
        with open(file_path, "w") as file:
            json.dump({}, file, indent=4)

    with open(file_path, "r") as file:
        data = json.load(file)

    data.update(new_dict)

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)