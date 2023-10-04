cd ../evaluation

##### chatgpt_2shot_code_use_caption_ocr #####
# generate code
python generate_response.py \
--model gpt-3.5-turbo \
--output_dir ../results/chatgpt \
--output_file output_chatgpt_2shot_code_use_caption_ocr.json \
--shot_num 2 \
--shot_type code \
--use_caption \
--use_ocr \
--caption_file ../data/texts/captions_bard.json \
--ocr_file ../data/texts/ocrs_easyocr.json 

# extract answer
python extract_answer.py \
--output_dir ../results/chatgpt \
--output_file output_chatgpt_2shot_code_use_caption_ocr.json \
--response_label execution --rerun

# calculate score
python calculate_score.py \
--output_dir ../results/chatgpt \
--output_file output_chatgpt_2shot_code_use_caption_ocr.json \
--score_file scores_chatgpt_2shot_code_use_caption_ocr.json
