set -x
bash run_base_gen_prompt_enron.sh $1
python prompt_matching.py -f $1