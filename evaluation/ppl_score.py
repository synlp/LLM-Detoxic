import argparse
import json
import math
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def rtp_PPL(results, model, tokenizer):
    
    n = len(results)
    pbar = tqdm(total=n, desc="Calculating PPL")
    losses = []
    for i in range(n):
        prompt = results[i]['prompt']
        response = results[i]['response']
        
        p_input_ids = tokenizer(prompt).input_ids
        
        if prompt[-1] != ' ':
            text = prompt + ' ' + response
        else:
            text = prompt + response
        
        input_ids = tokenizer(text, return_tensors='pt').input_ids.to(model.device)
        
        labels = input_ids.clone()
        labels[0, :len(p_input_ids)] = -100
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
        
        loss = outputs.loss.item() 
        losses.append(loss)
        pbar.update(1)
    
    pbar.close()
  
    mean_loss = sum(losses) / len(losses)
    ppl = math.exp(mean_loss)
    return ppl

def alpha_range_rtp_ppl(range_results, ppl_model, ppl_tokenizer):
  
    scores = []
    for pair in range_results:
        alpha = pair['alpha']
        print(f'alpha={alpha}, PPL测试开始')
        results = pair['prompt-response pairs']
        score = rtp_PPL(results, ppl_model, ppl_tokenizer)
        scores.append({'alpha': alpha, 'PPL': score})
        print(f'alpha={alpha}, PPL测试完毕: {score}')
    return scores

def main(args):
    
    print(f"Loading PPL model from {args.model_path}...")
    ppl_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16 
    )
    ppl_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    if ppl_tokenizer.pad_token is None:
        ppl_tokenizer.pad_token = ppl_tokenizer.eos_token
        ppl_tokenizer.pad_token_id = ppl_tokenizer.eos_token_id

    with open(args.input_json, 'r') as f:
        data = json.load(f)
    
    ppl_scores = alpha_range_rtp_ppl(data, ppl_model, ppl_tokenizer)
    
    print("\nPPL Results:")
    for score in ppl_scores:
        print(f"alpha={score['alpha']}: PPL={score['PPL']:.4f}")
    
    output_path = args.input_json.replace('.json', '_ppl.json')
    with open(output_path, 'w') as f:
        json.dump(ppl_scores, f, indent=4)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='计算模型生成的PPL分数')
    parser.add_argument('--model_path', type=str, required=True,
                        help='llama2-13b模型路径')
    parser.add_argument('--input_json', type=str, required=True,
                        help='包含生成结果的JSON文件路径')
    
    args = parser.parse_args()
    main(args)
