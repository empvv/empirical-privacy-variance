set -x
bash run_base_gen_prompt_enron_random.sh $1
python prompt_matching_random.py -f $1