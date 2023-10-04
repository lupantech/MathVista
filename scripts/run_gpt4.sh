cd ../evaluation

##### zero-shot gpt4 #####
# generate solution
python generate_response.py \
--model gpt-4-0613 \
--output_dir ../results/gpt4 \
--output_file output_gpt4.json

# extract answer
python extract_answer.py \
--output_dir ../results/gpt4 \
--output_file output_gpt4.json

# calculate score
python calculate_score.py \
--output_dir ../results/gpt4 \
--output_file output_gpt4.json \
--score_file scores_gpt4.json
