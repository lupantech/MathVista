import argparse
import os
import re
import sys

from openai import AzureOpenAI
from tqdm import tqdm

sys.path.append('../')
from models import gpt
from prompts.ext_ans import demo_prompt
from utilities import get_chat_response, read_json, save_json

def verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == "" or extraction == None:
        return False
    return True


def create_test_prompt(demo_prompt, query, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"{query}\n\n{response}"
    full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
    return full_prompt


def extract_answer(model, response, problem, quick_extract=False):
    question_type = problem['question_type']
    answer_type = problem['answer_type']
    choices = problem['choices']
    query = problem['query']
    pid = problem['pid']

    if response == "":
        return ""

    if question_type == 'multi_choice' and response in choices:
        return response

    if answer_type == "integer":
        try:
            extraction = int(response)
            return str(extraction)
        except Exception as e:
            pass

    if answer_type == "float":
        try:
            extraction = str(float(response))
            return extraction
        except Exception as e:
            pass

    # quick extraction
    if quick_extract:
        print("Quickly extracting answer...")
        # The answer is "text". -> "text"
        try:
            result = re.search(r'The answer is "(.*)"\.', response)
            if result:
                extraction = result.group(1)
                return extraction
        except Exception as e:
            pass

    # general extraction
    try:
        full_prompt = create_test_prompt(demo_prompt, query, response)
        response = model.get_response(user_prompt=full_prompt)
        return response
    except Exception as e:
        print(e)
        print(f"Error in extracting answer for {pid}")

    return ""


def parse_args():
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--results_file_path', type=str, default='answer.json')
    parser.add_argument('--response_label', type=str, default='response', help='response label for the input file')
    parser.add_argument('--number', type=int, default=-1, help='number of problems to run')
    parser.add_argument('--quick_extract', action='store_true', help='use rules to extract answer for some problems')
    parser.add_argument('--rerun', action='store_true', help='rerun the answer extraction')
    # output
    parser.add_argument('--save_every', type=int, default=10, help='save every n problems')

    parser.add_argument('--azure_openai_api_endpoint', type=str, default=os.getenv("AZURE_OPENAI_API_ENDPOINT"))
    parser.add_argument('--azure_openai_api_key', type=str, default=os.getenv("AZURE_OPENAI_API_KEY"))
    parser.add_argument('--azure_openai_api_version', type=str, default=os.getenv("AZURE_OPENAI_API_VERSION"))
    parser.add_argument('--azure_openai_model', type=str, default=os.getenv("AZURE_OPENAI_MODEL"))

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # args
    label = args.response_label

    assert (
        args.azure_openai_api_endpoint is not None
    ), "Env var AZURE_OPENAI_API_ENDPOINT is not set but is required for OpenAI client."
    assert (
        args.azure_openai_api_key is not None
    ), "Env var AZURE_OPENAI_API_KEY is not set but is required for OpenAI client."
    assert (
        args.azure_openai_api_version is not None
    ), "Env var AZURE_OPENAI_API_VERSION is not set but is required for OpenAI client."
    assert args.azure_openai_model is not None, "Env var AZURE_OPENAI_MODEL is not set but is required for OpenAI client."

    client = AzureOpenAI(
        azure_endpoint=args.azure_openai_api_endpoint,
        api_key=args.azure_openai_api_key,
        api_version=args.azure_openai_api_version,
    )
    model = gpt.GPT_Model(client=client, model=args.azure_openai_model)

    print(f"Reading {args.results_file_path}...")
    results = read_json(args.results_file_path)

    full_pids = list(results.keys())
    if args.number > 0:
        full_pids = full_pids[:min(args.number, len(full_pids))]
    print("Total Number of testing problems:", len(full_pids))

    skip_pids = []
    for pid, problem in results.items():
        extraction = problem.get('extraction')
        if extraction is not None and verify_extraction(extraction):
            skip_pids.append(problem['pid'])

    if args.rerun:
        test_pids = full_pids
    else:
        if len(skip_pids) > 0:
            print("Removing problems with existing valid response...")
        test_pids = [pid for pid in full_pids if pid not in skip_pids]

    print("Number of test problems to run:", len(test_pids))

    for i, pid in enumerate(tqdm(test_pids)):
        problem = results[pid]

        assert label in problem
        response = problem[label]
        extraction = extract_answer(model, response, problem, args.quick_extract)
        results[pid]['extraction'] = extraction

        if i % args.save_every == 0 or i == len(test_pids) - 1:
            save_json(results, args.results_file_path)
            print(f"Saved results to {args.results_file_path}")


if __name__ == '__main__':
    main()
