import argparse
import json
import numpy as np
from tqdm import tqdm
from detoxify import Detoxify

def rtp_tox(results, device):
    
    toxicity_scores = []
    scorer = Detoxify('original', device=device)
    pbar = tqdm(total=len(results), desc="Calculating toxicity scores")
    for pair in results:
        toxicity_scores.append(scorer.predict(pair['response'])['toxicity'])
        pbar.update(1)
    pbar.close()
    return np.mean(toxicity_scores)

def alpha_range_rtp_tox(pair_results, device):
    
    scores = []
    for pair in pair_results:
        alpha = pair['alpha']
        print(f'alpha={alpha}, 毒性分数测试开始')
        results = pair['prompt-response pairs']
        score = rtp_tox(results, device)
        scores.append({'alpha': alpha, 'toxicity_score': score})
        print(f'alpha={alpha}, 毒性分数测试完毕: {score}')
    return scores

def main(args):
    
    with open(args.input_json, 'r') as f:
        data = json.load(f)
    
    toxicity_scores = alpha_range_rtp_tox(data, args.device)   
    
    print("\n毒性分数结果:")
    for score in toxicity_scores:
        print(f"alpha={score['alpha']}: toxicity_score={score['toxicity_score']:.4f}")
    
    output_path = args.input_json.replace('.json', '_toxicity.json')
    with open(output_path, 'w') as f:
        json.dump(toxicity_scores, f, indent=4)
    print(f"\n结果已保存到: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='计算模型生成的毒性分数')
    parser.add_argument('--device', type=str, required=True, 
                        help='运行设备 (如 cuda:0 或 cpu)')
    parser.add_argument('--input_json', type=str, required=True,
                        help='包含生成结果的JSON文件路径')
    
    args = parser.parse_args()
    main(args)
