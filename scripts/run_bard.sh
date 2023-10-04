cd ../evaluation

##### multimodal-bard #####
# generate solution
python generate_response.py \
--model bard \
--output_dir ../results/bard \
--output_file output_bard.json

# extract answer
python extract_answer.py \
--output_dir ../results/bard \
--output_file output_bard.json 

# calculate score
python calculate_score.py \
--output_dir ../results/bard \
--output_file output_bard.json \
--score_file scores_bard.json
