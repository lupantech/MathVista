import argparse
import io
import os
import sys

from openai import AzureOpenAI
from tqdm import tqdm

sys.path.append('../')
from build_query import create_query_data
from utilities import read_json, save_json

# from models import bard
# from models import claude
from models import gpt


def verify_response(response):
    if isinstance(response, str):
        response = response.strip()
    if response == "" or response == None:
        return False
    if "Response Error" in response:
        return False
    return True


def evaluate_code(code_string):
    # execute_code_and_capture_output
    # Backup the original stdout
    old_stdout = sys.stdout

    # Redirect stdout to capture the output
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    # Try executing the code and capture any exception
    error = None
    try:
        exec(code_string)
    except Exception as e:
        error = e

    # Restore the original stdout
    sys.stdout = old_stdout

    # Get the captured output
    captured_output = new_stdout.getvalue()
    if isinstance(captured_output, str):
        captured_output = captured_output.strip()

    # Return the captured output or error
    return captured_output, error


def parse_args():
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--input_file', type=str, default='testmini.json')
    # output
    parser.add_argument('--output_dir', type=str, default='../results/bard')
    parser.add_argument('--output_file', type=str, default='output_bard.json')
    # model
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='llm engine',
                        choices = ['gpt-3.5-turbo', 'claude-2', 'gpt4', 'gpt-4-0613', 'bard'])
    parser.add_argument('--key', type=str, default='', help='key for llm api')
    # query
    parser.add_argument('--query_file', type=str, default=None)
    parser.add_argument('--caption_file', type=str, default='../data/texts/captions_bard.json')
    parser.add_argument('--ocr_file', type=str, default='../data/texts/ocrs_easyocr.json')
    parser.add_argument('--shot_type', type=str, default='solution', help='shot type',
                        choices = ['solution', 'code'])
    parser.add_argument('--shot_num', type=int, default=0, help='number of shot examples')
    parser.add_argument('--use_caption', action='store_true', help='use caption data')
    parser.add_argument('--use_ocr', action='store_true', help='use ocr data')
    # other settings
    parser.add_argument('--rerun', action='store_true', help='rerun answer extraction for all problems')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--azure_openai_api_endpoint', type=str, default=os.getenv("AZURE_OPENAI_API_ENDPOINT"))
    parser.add_argument('--azure_openai_api_key', type=str, default=os.getenv("AZURE_OPENAI_API_KEY"))
    parser.add_argument('--azure_openai_api_version', type=str, default=os.getenv("AZURE_OPENAI_API_VERSION"))
    parser.add_argument('--azure_openai_model', type=str, default=os.getenv("AZURE_OPENAI_MODEL"))
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load data
    input_file = os.path.join(args.data_dir, args.input_file)
    print(f"Reading {input_file}...")
    data = read_json(input_file)

    # load or create query data
    if args.query_file:
        query_file = os.path.join(args.data_dir, args.query_file)
        if os.path.exists(query_file):
            print(f"Loading existing {query_file}...")
            query_data = read_json(query_file)
    else:
        print("\nCreating new query...")
        # load caption
        caption_data = {}
        if args.use_caption:
            caption_file = args.caption_file
            if os.path.exists(caption_file):
                print(f"Reading {caption_file}...")
                try:
                    caption_data = read_json(caption_file)["texts"]
                    print("Caption data loaded.")
                except:
                    print("Caption data not found!! Please Check.")
        # load ocr
        ocr_data = {}
        if args.use_ocr:
            ocr_file = args.ocr_file
            if os.path.exists(ocr_file):
                print(f"Reading {ocr_file}...")
                try:
                    ocr_data = read_json(ocr_file)["texts"]
                    print("OCR data loaded.")
                except:
                    print("OCR data not found!! Please Check.")
        # create query
        query_data = create_query_data(data, caption_data, ocr_data, args)

    # output file
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, args.output_file)

    # load results
    if os.path.exists(output_file):
        print("Results already exist.")
        print(f"Reading {output_file}...")
        results = read_json(output_file)
    else:
        results = {}

    # load model
    model_name = args.azure_openai_model if args.azure_openai_model else args.model
    print(f"\nLoading {model_name}...")
    if model_name == 'bard':
        if args.key == '':
            print("Loading key from environment variable")
            key = os.environ['_BARD_API_KEY']
        else:
            key = args.key
        model = bard.Bard_Model(key)

    elif "gpt" in model_name:
        key = args.azure_openai_api_key if args.azure_openai_api_key else args.key
        if key == '':
            key = os.getenv("OPENAI_API_KEY")

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

        model = gpt.GPT_Model(client=client, model=model_name)

    elif "claude" in model_name:
        if args.key == '':
            print("Loading token from environment variable")
            key = os.environ.get("ANTHROPIC_API_KEY")
        else:
            key = args.key
        model = claude.Claude_Model(model_name, key)

    print(f"Model loaded.")

    # build final test pid list
    test_pids = list(data.keys())
    print("\nNumber of test problems in total:", len(test_pids))

    skip_pids = []
    if not args.rerun:
        print("\nRemoving problems with existing valid response...")
        for problem_id in test_pids:
            # print(f"Checking {pid}...")
            if problem_id in results and 'response' in results[problem_id]:
                response = results[problem_id]['response']
                if verify_response(response):
                    # print(f"Valid response found for {pid}.")
                    skip_pids.append(problem_id)

    if len(skip_pids) > 0:
        print(f"Found existing results file with {len(skip_pids)} problems with valid responses. Skipping these problems...")
    else:
        print("Rerun answer extraction for all problems...")

    test_pids = [pid for pid in test_pids if pid not in skip_pids]
    print("Number of test problems to run:", len(test_pids))
    # print(test_pids)

    # tqdm, enumerate results
    for problem_index, problem_id in enumerate(tqdm(test_pids)):
        problem = data[problem_id]
        query = query_data[problem_id]
        image = problem['image']
        image_path = os.path.join(args.data_dir, image)

        if args.debug:
            print("--------------------------------------------------------------")
            print(f"Generating response for index: {problem_index} id: {problem_id}...")
        try:
            response = model.get_response(user_prompt=query)
            # print(f"Response: {response}")
            results[problem_id] = problem
            results[problem_id]['query'] = query
            if args.shot_type == 'solution':
                results[problem_id]['response'] = response
            else:
                output, error = evaluate_code(response)
                results[problem_id]['response'] = response
                results[problem_id]['execution'] = output
                results[problem_id]['error'] = str(error)
            if args.debug:
                print(f"\n#Query: \n{query}")
                print(f"\n#Response: \n{response}")
        except Exception as e:
            print(e)
            print(f"Error in extracting answer for {problem_id}")
            results[problem_id]['error'] = e

        try:
            if args.debug:
                print(f"Saving results to {output_file}...")
            save_json(results, output_file)
            if args.debug:
                print(f"Results saved.")
        except Exception as e:
            print(e)
            print(f"Error in saving {output_file}")


if __name__ == '__main__':
    main()
