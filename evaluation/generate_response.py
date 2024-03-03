import argparse
import io
import logging
import os
import sys

from datasets import load_dataset
from openai import AzureOpenAI
from rich.logging import RichHandler
from tqdm import tqdm

from evaluation.build_query import create_query_data
from utilities import read_json, save_json


def verify_response(response):
    if isinstance(response, str):
        response = response.strip()
    if response == "" or response is None:
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
    parser.add_argument('--dataset_name', type=str, default='AI4Math/MathVista')
    parser.add_argument('--test_split_name', type=str, default='testmini')
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--input_file', type=str, default='testmini.json')
    # output
    parser.add_argument('--output_dir', type=str, default='../results/bard')
    parser.add_argument('--output_file', type=str, default='output_bard.json')
    parser.add_argument('--max_num_problems', type=int, default=-1, help='The number of problems to run')
    parser.add_argument('--save_every', type=int, default=100, help='save every n problems')
    # Local Model
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    # Remote model
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-3.5-turbo',
        help='llm engine',
        choices=['gpt-3.5-turbo', 'claude-2', 'gpt4', 'gpt-4-0613', 'bard'],
    )
    parser.add_argument('--key', type=str, default='', help='key for llm api')
    # query
    parser.add_argument('--query_file', type=str, default=None)
    parser.add_argument('--caption_file', type=str, default='../data/texts/captions_bard.json')
    parser.add_argument('--ocr_file', type=str, default='../data/texts/ocrs_easyocr.json')
    parser.add_argument('--shot_type', type=str, default='solution', help='shot type', choices=['solution', 'code'])
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
    logging.info("MathVista: Generating Responses - Start")
    args = parse_args()

    # load data
    logging.info(f"Loading dataset {args.dataset_name}, split {args.test_split_name}...")
    data_list = load_dataset(args.dataset_name, split=args.test_split_name)
    # Convert Hugging Face data into dictionary to match local data format
    # TODO: Convert scripts not to depend on dictionary .json format. Update to use .jsonl format
    data = {item['pid']: item for item in data_list}

    # load or create query data
    if args.query_file:
        query_file = os.path.join(args.data_dir, args.query_file)
        if os.path.exists(query_file):
            logging.info(f"Loading existing {query_file}...")
            query_data = read_json(query_file)
    else:
        logging.info("Creating new query...")

        caption_data = {}
        if args.use_caption:
            caption_file = args.caption_file
            if os.path.exists(caption_file):
                logging.info(f"Reading {caption_file}...")
                try:
                    caption_data = read_json(caption_file)["texts"]
                    logging.info("Caption data loaded.")
                except Exception as e:
                    logging.info("Caption data not found!! Please Check.")

        ocr_data = {}
        if args.use_ocr:
            ocr_file = args.ocr_file
            if os.path.exists(ocr_file):
                logging.info(f"Reading {ocr_file}...")
                try:
                    ocr_data = read_json(ocr_file)["texts"]
                    logging.info("OCR data loaded.")
                except Exception as e:
                    logging.info("OCR data not found!! Please Check.")

        query_data = create_query_data(data, caption_data, ocr_data, args)

    # If we were given a custom model path, load that model, otherwise use a remote service model
    if args.model_path:
        # from models import llava

        logging.info(f"Loading model from {args.model_path}...")
        # TODO: Add support for local models
        raise NotImplementedError("Local models are not yet supported.")
    else:
        model_name = args.azure_openai_model if args.azure_openai_model else args.model
        logging.info(f"Loading {model_name}...")

        if model_name == 'bard':
            from models import bard

            if args.key == '':
                logging.info("Loading key from environment variable")
                key = os.environ['_BARD_API_KEY']
            else:
                key = args.key
            model = bard.Bard_Model(key)
        elif "gpt" in model_name:
            from models import gpt

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
            assert (
                args.azure_openai_model is not None
            ), "Env var AZURE_OPENAI_MODEL is not set but is required for OpenAI client."

            client = AzureOpenAI(
                azure_endpoint=args.azure_openai_api_endpoint,
                api_key=args.azure_openai_api_key,
                api_version=args.azure_openai_api_version,
            )

            model = gpt.GPT_Model(client=client, model=model_name)

        elif "claude" in model_name:
            from models import claude

            if args.key == '':
                logging.info("Loading token from environment variable")
                key = os.environ.get("ANTHROPIC_API_KEY")
            else:
                key = args.key
            model = claude.Claude_Model(model_name, key)
        else:
            raise ValueError(f"Model {model_name} not supported.")

    logging.info(f"Model loaded.")

    full_pids = list(data.keys())

    os.makedirs(args.output_dir, exist_ok=True)
    output_file_path = os.path.join(args.output_dir, args.output_file)

    # load results
    if os.path.exists(output_file_path):
        logging.info("Results already exist.")
        logging.info(f"Reading {output_file_path}...")
        results = read_json(output_file_path)
    else:
        results = {}

    skip_pids = []
    if not args.rerun:
        for problem_id in full_pids:
            # logging.info(f"Checking {pid}...")
            if problem_id in results and 'response' in results[problem_id]:
                response = results[problem_id]['response']
                if verify_response(response):
                    # logging.info(f"Valid response found for {pid}.")
                    skip_pids.append(problem_id)

    if len(skip_pids) > 0:
        logging.info(
            f"Found existing results file with {len(skip_pids)} problems with valid responses. Skipping these problems..."
        )

    test_pids = [pid for pid in full_pids if pid not in skip_pids]

    if args.max_num_problems > 0:
        test_pids = test_pids[: min(args.max_num_problems, len(test_pids))]
        logging.warning(f'Limiting number of problems to {args.max_num_problems}.')

    logging.info(f"Number of test problems to run: {len(test_pids)}")

    for i, problem_id in enumerate(tqdm(test_pids)):
        problem: dict = data[problem_id].copy()

        # Remove decoded Image for JSON deserialization
        problem_decoded_image = problem['decoded_image']
        problem.pop('decoded_image')

        query = query_data[problem_id]

        logging.debug("--------------------------------------------------------------")
        logging.debug(f"Generating response for problem: {problem_id}...")
        try:
            response = model.get_response(user_prompt=query, decoded_image=problem_decoded_image)
            results[problem_id] = problem
            results[problem_id]['query'] = query
            if args.shot_type == 'solution':
                results[problem_id]['response'] = response
            else:
                output, error = evaluate_code(response)
                results[problem_id]['response'] = response
                results[problem_id]['execution'] = output
                results[problem_id]['error'] = str(error)
            logging.debug(f"Query: \n{query}")
            logging.debug(f"Response: \n{response}")
        except Exception as e:
            logging.error(f"Error in extracting answer for {problem_id}")
            logging.error(e)
            results[problem_id] = problem
            results[problem_id]['error'] = str(e)

        if (i % args.save_every == 0 and i > 0) or i == len(test_pids) - 1:
            try:
                save_json(results, output_file_path)
                logging.info(f"Saved results to {output_file_path}")
            except Exception as e:
                logging.info(f"Error in saving {output_file_path}")
                logging.info(e)

    logging.info("MathVista: Generating Responses - Finish")


if __name__ == '__main__':
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                markup=False,
                show_path=False,
                omit_repeated_times=False,
            )
        ],
    )
    logger_blocklist = [
        "asyncio",
        "azure",
        "azureml",
        "datasets",
        "httpx",
        "httpcore",
        "filelock",
        "fsspec",
        "msal",
        "msrest",
        "openai",
        "PIL",
        "urllib3",
    ]
    for module in logger_blocklist:
        logging.getLogger(module).setLevel(logging.WARNING)

    main()
