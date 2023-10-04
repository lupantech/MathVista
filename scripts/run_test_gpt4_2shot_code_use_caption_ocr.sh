cd ../evaluation

#### output_gpt4_2shot_code_use_caption_ocr #####
# generate code
# OPENAI_API_KEY="" 
python generate_response.py \
--model gpt-4-0613 \
--input_file test.json \
--output_dir ../results/test_gpt4 \
--output_file output_test_gpt4_2shot_code_use_caption_ocr.json \
--shot_num 2 \
--shot_type code \
--use_caption \
--use_ocr \
--caption_file ../data/texts/captions_bard.json \
--ocr_file ../data/texts/ocrs_easyocr.json 

# extract answer
python extract_answer.py \
--output_dir ../results/test_gpt4 \
--output_file output_test_gpt4_2shot_code_use_caption_ocr.json \
--response_label execution

# calculate score
python calculate_score.py \
--gt_file ../data_final/private/data.json \
--output_dir ../results/test_gpt4 \
--output_file output_test_gpt4_2shot_code_use_caption_ocr.json \
--score_file scores_test_gpt4_2shot_code_use_caption_ocr.json
