cd ../evaluation

##### llavar #####
# extract answer
python extract_answer.py \
--output_file llavar/output_llavar.json \
--response_label LLaVar

# calculate score
python calculate_score.py \
--output_dir ../results/llavar \
--output_file output_llavar.json \
--score_file scores_llavar.json
