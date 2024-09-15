import json
import os
import pickle
import re

import cv2
import PIL.Image as Image
from word2number import w2n


def create_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def read_csv(file):
    data = []
    with open(file, 'r') as f:
        for line in f:
            data.append(line.strip())
    return data


def read_pandas_csv(csv_path):
    # read a pandas csv sheet
    import pandas as pd

    df = pd.read_csv(csv_path)
    return df


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_jsonl(file):
    with open(file, 'r') as f:
        data = [json.loads(line) for line in f]
    return data


def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        data_json = json.dumps(data, indent=4)
        f.write(data_json)


def save_array_img(path, image):
    cv2.imwrite(path, image)


def contains_digit(text):
    # check if text contains a digit
    if any(char.isdigit() for char in text):
        return True
    return False


def contains_number_word(text):
    # check if text contains a number word
    ignore_words = ["a", "an", "point"]
    words = re.findall(r'\b\w+\b', text)  # This regex pattern matches any word in the text
    for word in words:
        if word in ignore_words:
            continue
        try:
            w2n.word_to_num(word)
            return True  # If the word can be converted to a number, return True
        except ValueError:
            continue  # If the word can't be converted to a number, continue with the next word

    # check if text contains a digit
    if any(char.isdigit() for char in text):
        return True

    return False  # If none of the words could be converted to a number, return False


def contains_quantity_word(text, special_keep_words=[]):
    # check if text contains a quantity word
    quantity_words = [
        "most",
        "least",
        "fewest" "more",
        "less",
        "fewer",
        "largest",
        "smallest",
        "greatest",
        "larger",
        "smaller",
        "greater",
        "highest",
        "lowest",
        "higher",
        "lower",
        "increase",
        "decrease",
        "minimum",
        "maximum",
        "max",
        "min",
        "mean",
        "average",
        "median",
        "total",
        "sum",
        "add",
        "subtract",
        "difference",
        "quotient",
        "gap",
        "half",
        "double",
        "twice",
        "triple",
        "square",
        "cube",
        "root",
        "approximate",
        "approximation",
        "triangle",
        "rectangle",
        "circle",
        "square",
        "cube",
        "sphere",
        "cylinder",
        "cone",
        "pyramid",
        "multiply",
        "divide",
        "percentage",
        "percent",
        "ratio",
        "proportion",
        "fraction",
        "rate",
    ]

    quantity_words += special_keep_words  # dataset specific words

    words = re.findall(r'\b\w+\b', text)  # This regex pattern matches any word in the text
    if any(word in quantity_words for word in words):
        return True

    return False  # If none of the words could be converted to a number, return False


def is_bool_word(text):
    if text in ["Yes", "No", "True", "False", "yes", "no", "true", "false", "YES", "NO", "TRUE", "FALSE"]:
        return True
    return False


def is_digit_string(text):
    # remove ".0000"
    text = text.strip()
    text = re.sub(r'\.0+$', '', text)
    try:
        int(text)
        return True
    except ValueError:
        return False


def is_float_string(text):
    # text is a float string if it contains a "." and can be converted to a float
    if "." in text:
        try:
            float(text)
            return True
        except ValueError:
            return False
    return False


def copy_image(image_path, output_image_path):
    from shutil import copyfile

    copyfile(image_path, output_image_path)


def copy_dir(src_dir, dst_dir):
    from shutil import copytree

    # copy the source directory to the target directory
    copytree(src_dir, dst_dir)


def get_image_size(img_path):
    img = Image.open(img_path)
    width, height = img.size
    return width, height
