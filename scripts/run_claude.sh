cd ../evaluation

##### zero-shot claude #####
# generate solution
python generate_response.py \
--model claude-2 \
--output_dir ../results/claude \
--output_file output_claude.json

# extract answer
python extract_answer.py \
--output_dir ../results/claude \
--output_file output_claude.json

# calculate score
python calculate_score.py \
--output_dir ../results/claude \
--output_file output_claude.json \
--score_file scores_claude.json
