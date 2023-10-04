cd ../evaluation

##### llava-llama-2-13b #####
# extract answer
python extract_answer.py \
--output_file llava-llama-2-13b/output_llava_llama_2_13b.json \
--response_label llava_llama_2_13b

# calculate score
python calculate_score.py \
--output_dir ../results/llava-llama-2-13b \
--output_file output_llava_llama_2_13b.json \
--score_file scores_llava_llama_2_13b.json
