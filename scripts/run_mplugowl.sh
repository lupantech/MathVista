cd ../evaluation

#####  mplugowl-ft-7b #####
# extract answer
python extract_answer.py \
--output_file mplugowl-ft-7b/output_mplugowl_7b_ft.json \
--response_label mplugowl

# calculate score
python calculate_score.py \
--output_dir ../results/mplugowl-ft-7b \
--output_file output_mplugowl_7b_ft.json \
--score_file scores_mplugowl_7b_ft.json \
