cd ../evaluation

##### chatgpt_2shot_code #####
# generate code
python generate_response.py \
--model gpt-3.5-turbo \
--output_dir ../results/chatgpt \
--output_file output_chatgpt_2shot_code.json \
--shot_num 2 \
--shot_type code

# extract answer
python extract_answer.py \
--output_dir ../results/chatgpt \
--output_file output_chatgpt_2shot_code.json

# calculate score
python calculate_score.py \
--output_dir ../results/chatgpt \
--output_file output_chatgpt_2shot_code.json \
--score_file scores_chatgpt_2shot_code.json
