cd ../evaluation

##### zero-shot chatgpt #####
# generate solution
python generate_response.py \
--model gpt-3.5-turbo \
--output_dir ../results/chatgpt \
--output_file output_chatgpt.json

# extract answer
python extract_answer.py \
--output_dir ../results/chatgpt \
--output_file output_chatgpt.json

# calculate score
python calculate_score.py \
--output_dir ../results/chatgpt \
--output_file output_chatgpt.json \
--score_file scores_chatgpt.json
