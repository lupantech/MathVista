cd ../evaluation

##### minigpt4 #####
# extract answer
python extract_answer.py \
--output_file minigpt4/output_mathvista_minigpt4_llama2.json \
--response_label minigpt4

# calculate score
python calculate_score.py \
--output_dir ../results/minigpt4 \
--output_file output_mathvista_minigpt4_llama2.json \
--score_file scores_mathvista_minigpt4_llama2.json \
