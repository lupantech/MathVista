"""
Reference: https://bard.google.com/
"""

import os
import json
import tqdm
import argparse

import sys
sys.path.append('../')
from utilities import *

from models import bard

def verify_response(response):
    if isinstance(response, str):
        response = response.strip() 
    if response == "" or response == None:
        return False
    if "Response Error" in response:
        return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='../data')
    parser.add_argument('--output_dir', type=str, default='../data/texts')
    parser.add_argument('--output_file', type=str, default='captions_bard.json')
    parser.add_argument('--model', type=str, default='bard', help='model name')
    parser.add_argument('--key', type=str, default='', help='key for bard api')
    args = parser.parse_args()

    # data_files = ["testmini.json", "test.json"]
    data_files = ["testmini.json"]

    data = {}
    for data_file in data_files:
        d = json.load(open(os.path.join(args.output_dir, data_file), 'r'))
        data.update(d)

    pids = list(data.keys())
    print("number of images: ", len(pids))

    ### Read skipped image ids
    result_file = os.path.join("../results/bard/output_bard.json")
    results = json.load(open(result_file, 'r'))
    skipped = []
    for pid in results:
        response = results[pid]['response']
        if "I can't help with images of people" in response:
            skipped.append(pid)
    print("number of skipped images: ", len(skipped))

    # output file
    output_file = os.path.join(args.output_dir, args.output_file)
    
    # final test pids
    if os.path.exists(output_file):
        print("Loading existing results...")
        results = json.load(open(output_file, 'r'))
    else:
        results = {
            "model": "bard",
            "url": "https://bard.google.com/",
            "date": "2023-09-13",
            "texts": {}
        }

    test_pids = []
    texts = results['texts']

    for pid in pids:
        if pid in skipped:
            continue
        if pid in texts:
            response = texts[pid]
            if not verify_response(response):
                test_pids.append(pid)
        else:
            test_pids.append(pid)

    print("\nfinal number of images to run: ", len(test_pids))

    # build model
    query = "Describe the fine-grained content of the image or figure, including scenes, objects, relationships, and any text present."
    
    if args.model == 'bard':
        if args.key == '':
            print("Loading key from environment variable")
            key = os.environ['_BARD_API_KEY']
        else:
            key = args.key
        model = bard.Bard_Model(key)

    for pid in tqdm.tqdm(test_pids):
        image_path = os.path.join(args.image_dir, data[pid]["image"])

        print(f"Generating response for {pid}...")
        try:
            response = model.get_response(image_path, query)
            # print(f"Response: {response}")
            results['texts'][pid] = response
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            print(e)


    