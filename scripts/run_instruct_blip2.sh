cd ../evaluation

##### instruct-blip2 #####
# extract answer
python extract_answer.py \
--output_file instruct-blip2-vicuna13b/output_instructblip2.json \
--response_label instructblip2

# calculate score
python calculate_score.py \
--output_dir ../results/instruct-blip2-vicuna13b \
--output_file output_instructblip2.json \
--score_file scores_instructblip2.json \
