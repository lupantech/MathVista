cd ../evaluation

### llava-llama-2-13b #####
# extract answer
python extract_answer.py \
--output_file test_llava/output_llava_llama_2_13b.json

# calculate score
python calculate_score.py \
--gt_file ../data_final/private/data.json \
--output_dir ../results/test_llava \
--output_file output_llava_llama_2_13b.json \
--score_file scores_test_llava_llama_2_13b.json
