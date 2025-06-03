import sys
import os
import json
import torch
import argparse
from tqdm import tqdm
import random
import numpy as np
from transformers import AutoTokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from modeling import LlamaForCausalLMTeacher, LlamaForCausalLMTarget

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    
    A = torch.load(args.matrix_a)
    if 'mapping' in A:
        A = A['mapping']
    
    
    teacher = LlamaForCausalLMTeacher.from_pretrained(args.teacher_model)
    target = LlamaForCausalLMTarget.from_pretrained(args.target_model)
    
    device = torch.device(args.device)
    teacher.to(device)
    target.to(device)
    A = A.to(device)

    teacher.eval()
    target.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model,
        max_length=512,
        return_tensors='pt',
        padding=True,
        truncation=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    with open(args.data_path, 'r') as f:
        challenge_prompts = [json.loads(line)['prompt'] for line in f]
    
    n = len(challenge_prompts)
    results = []
    
    target.get_teacher(teacher)
    target.use_alignment(A)
    
    set_seed(2025)
    
    for alpha in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        print(f'alpha={alpha}, start')
        pbar = tqdm(total=n)
        pr = []
        
        target.get_alpha(alpha)
        
        for prompt in challenge_prompts:
            tp = tokenizer(prompt, return_tensors='pt')
            with torch.no_grad():
                response = target.generate(
                    input_ids=tp.input_ids.to(device),
                    attention_mask=tp.attention_mask.to(device),
                    max_new_tokens=10,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            response_tokens = response[0, len(tp.input_ids[0]):].tolist()
            response_text = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            pr.append({'prompt': prompt, 'response': response_text})
            pbar.update(1)
        
        results.append({'alpha': alpha, 'prompt-response pairs': pr})
        print(f'alpha={alpha}, end')
    
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, required=True, 
                        help="Path to teacher model")
    parser.add_argument("--target_model", type=str, required=True,
                        help="Path to target model")
    parser.add_argument("--matrix_a", type=str, required=True,
                        help="Path to alignment matrix A")
    parser.add_argument("--device", type=str, required=True,
                        help="Device to use (e.g., 'cuda:0')"),
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to evaluation data set"),
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save output JSON")
    
    args = parser.parse_args()
    main(args)
