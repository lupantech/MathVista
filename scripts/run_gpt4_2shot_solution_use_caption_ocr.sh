cd ../evaluation

##### gpt4_2shot_solution_use_caption_ocr #####
# generate solution
python generate_response.py \
--model gpt-4-0613 \
--output_dir ../results/gpt4 \
--output_file output_gpt4_2shot_solution_use_caption_ocr.json \
--shot_num 2 \
--shot_type solution \
--use_caption \
--use_ocr \
--caption_file ../data/texts/captions_bard.json \
--ocr_file ../data/texts/ocrs_easyocr.json 

# extract answer
python extract_answer.py \
--output_dir ../results/gpt4 \
--output_file output_gpt4_2shot_solution_use_caption_ocr.json

# calculate score
python calculate_score.py \
--output_dir ../results/gpt4 \
--output_file output_gpt4_2shot_solution_use_caption_ocr.json \
--score_file scores_gpt4_2shot_solution_use_caption_ocr.json

