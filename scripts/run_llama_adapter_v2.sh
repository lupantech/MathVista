cd ../evaluation

##### llama_adapter_v2 #####
# extract answer
python extract_answer.py \
--output_file llama_adapter_v2/output_mathvista_llama_adapter_v2.json \
--response_label llama_adapter

# calculate score
python calculate_score.py \
--output_dir ../results/llama_adapter_v2 \
--output_file output_mathvista_llama_adapter_v2.json \
--score_file scores_mathvista_llama_adapter_v2.json
