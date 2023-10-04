# <img src="https://mathvista.github.io/static/images/mathvista.png" alt="Logo" style="zoom:10%;" /> MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts



Code for the Paper "[MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts](https://arxiv.org/abs/2310.02255)".

:bell: If you have any questions or suggestions, please don't hesitate to let us know. You can directly email [Pan Lu](https://lupantech.github.io/) using the email address lupantech@gmail.com, comment on the [Twitter](https://twitter.com/lupantech), or post an issue on this repository.

[[Project Page](https://mathvista.github.io/)] [[Paper](https://arxiv.org/abs/2310.02255)]

<p align="center">
    <img src="assets/mathvista.png" width="15%"> <br>
  Tentative logo for <b>MathVista</b>.
</p>




## 💥 News 💥

- **[2023.10.03]** Our paper is now accessible at https://arxiv.org/abs/2310.02255.



## About MathVista

Although Large Language Models (LLMs) and Large Multimodal Models (LMMs) exhibit impressive skills in various domains, their ability for mathematical reasoning within visual contexts has not been formally examined. Equipping LLMs and LMMs with this capability is vital for general-purpose AI assistants and showcases promising potential in education, data analysis, and scientific discovery.

To bridge this gap, we present **MathVista**, a benchmark designed to amalgamate challenges from **diverse mathematical and visual tasks**. We first taxonomize the key task types, reasoning skills, and visual contexts from the literature to guide our selection from **28 existing math-focused and visual question answering datasets**. Then, **we construct three new datasets, IQTest, FunctionQA, and PaperQA**, to accommodate for missing types of visual contexts. The problems featured often require deep visual understanding beyond OCR or image captioning, and compositional reasoning with rich domain-specific tools, thus posing a notable challenge to existing models.

We conduct **a comprehensive evaluation of 11 prominent open-source and proprietary foundation models** (LLMs, LLMs augmented with tools, and LMMs), and **early experiments with GPT-4V**. The best-performing model, Multimodal Bard, achieves only **58%** of human performance (34.8% vs 60.3%), indicating ample room for further improvement. Given this significant gap, **MathVista** fuels future research in the development of general-purpose AI agents capable of tackling mathematically intensive and visually rich real-world tasks. Preliminary tests show that **MathVista** also presents challenges to GPT-4V, underscoring the benchmark's importance.

For more details, you can find our project page [here](https://mathvista.github.io/) and our paper [here](https://arxiv.org/pdf/2310.02255.pdf).



## Download the MathVista Dataset

You can download the MathVista dataset from [Google Drive](https://drive.google.com/file/d/1jX_nKaoDALEttiN1IR0dr89qLVt8yBkO/view) and store the data files in the `data` folder as following:

```sh
├── data
│   ├── annot_testmini.json
│   ├── images
│   ├── pids_UniGeo.json
│   ├── query.json
│   ├── README.md
│   ├── source.json
│   ├── test.json
│   ├── testmini.json
│   └── texts
```

The MathVista dataset will be available at [HuggingFace Datasets](https://huggingface.co/datasets/lupantech/MathVista) shortly! Stay tuned~



## 🐙 Requirements

- [OpenAI API key](https://platform.openai.com/account/api-keys)
- [Claude API Key](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)
- [Bard API Key](https://bard.google.com/)

Install the python dependencies if you would like to reproduce our results:

```
pip install openai
pip install anthropic
pip install bardapi
```



## Run Experiments on MathVista

### Multimodal Bard

If you have setted Multimodal Bard, you can run the following commands:

Generate the response:

```sh
cd evaluation

python generate_response.py \
--model bard \
--output_dir ../results/bard \
--output_file output_bard.json
```

Extract the short answer text for score calculation:

```sh
python extract_answer.py \
--output_dir ../results/bard \
--output_file output_bard.json 
```

Calculate the final score:

```sh
python calculate_score.py \
--output_dir ../results/bard \
--output_file output_bard.json \
--score_file scores_bard.json
```

### Chain-of-Thought GPT-4

Generate the response:

```sh
cd evaluation

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
```

Extract the short answer text for score calculation:

```sh
python extract_answer.py \
--output_dir ../results/gpt4 \
--output_file output_gpt4_2shot_solution_use_caption_ocr.json
```

Calculate the final score:

```sh
python calculate_score.py \
--output_dir ../results/gpt4 \
--output_file output_gpt4_2shot_solution_use_caption_ocr.json \
--score_file scores_gpt4_2shot_solution_use_caption_ocr.json
```

### Program-of-Thought GPT-4

Generate the response:

```sh
cd evaluation

python generate_response.py \
--model gpt-4-0613 \
--output_dir ../results/gpt4 \
--output_file output_gpt4_2shot_code_use_caption_ocr.json \
--shot_num 2 \
--shot_type code \
--use_caption \
--use_ocr \
--caption_file ../data/texts/captions_bard.json \
--ocr_file ../data/texts/ocrs_easyocr.json 
```

Extract the short answer text for score calculation:

```sh
python extract_answer.py \
--output_dir ../results/gpt4 \
--output_file output_gpt4_2shot_code_use_caption_ocr.json \
--response_label execution
```

Calculate the final score:

```sh
python calculate_score.py \
--output_dir ../results/gpt4 \
--output_file output_gpt4_2shot_code_use_caption_ocr.json \
--score_file scores_gpt4_2shot_code_use_caption_ocr.json
```

### More Models

To run more models, please check out the running scripts at [`scripts`](https://github.com/lupantech/MathVista/tree/main/scripts).



## :coffee: Stay Connected!

Fantastic! I'm always open to engaging discussions, collaborations, or even just sharing a virtual coffee. To get in touch, visit [Pan Lu](https://lupantech.github.io/)'s homepage for contact information.




## :white_check_mark: Cite

If you find **MathVista** useful for your your research and applications, please kindly cite using this BibTeX:

```latex
@article{lu2023mathvista,
  title={MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts},
  author={Lu, Pan and Bansal, Hritik and Xia, Tony and Liu, Jiacheng and Li, Chunyuan and Hajishirzi, Hannaneh and Cheng, Hao and Chang, Kai-Wei and Galley, Michel and Gao, Jianfeng},
  journal={arXiv preprint arXiv:2310.02255},
  year={2023}
}
```

