"""
Reference: https://github.com/JaidedAI/EasyOCR 
"""

# ! pip install easyocr

import os
import easyocr
import json
import tqdm

if __name__ == "__main__":

    data_files = ["../data/testmini.json", "../data/test.json"]
    image_dir = "../data"
    output_file = "../data/texts/ocrs_easyocr.json"

    data = {}
    for data_file in data_files:
        d = json.load(open(data_file, 'r'))
        data.update(d)

    pids = list(data.keys())
    print("number of images: ", len(pids))

    reader = easyocr.Reader(['en']) # ocr reader

    results = {
        "model": "easyocr",
        "url": "https://github.com/JaidedAI/EasyOCR",
        "version": "1.1.8",
        "date": "2023-04-06",
        "texts": {}
    }
    for pid in tqdm.tqdm(pids):
        image_file = os.path.join(image_dir, data[pid]["image"])

        try:
            result = reader.readtext(image_file)
        except Exception as e:
            print(image_file, e)
            result = ""
        
        results["texts"][pid] = str(result)
        # print(result)
        # break

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    