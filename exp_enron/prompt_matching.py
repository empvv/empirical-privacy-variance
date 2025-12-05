import argparse
import json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', '-f', type=str, required=True)
args = parser.parse_args()

gen_file = f'eval_results/enron/filtered_{args.model_name}/generated_prompt_greedy.json'
gt_file = 'secrets/secret_prompt.jsonl'

completions = json.load(open(gen_file))

gts = []
with open(gt_file) as f:
    for line in f:
        gts.append(json.loads(line)['target'])

match_list = []
for i, (gt, completion) in enumerate(zip(gts, completions)):
    match_list.append(completion.startswith(gt))
    
match_df = pd.DataFrame({'match': match_list})

outfile = f'eval_results/enron/filtered_{args.model_name}/generated_prompt_greedy_match.csv'

match_df.to_csv(outfile)