cd ../evaluation

##### idefics-9b-instruct #####
# extract answer
python extract_answer.py \
--output_file idefics-9b-instruct/output_idefics_9b_instruct.json \
--response_label idefics

# calculate score
python calculate_score.py \
--output_dir ../results/idefics-9b-instruct \
--output_file output_idefics_9b_instruct.json \
--score_file scores_idefics_9b_instruct.json \
