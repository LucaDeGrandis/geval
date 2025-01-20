import argparse
import re
from typing import List, Dict, Any
import json
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(description='Compute G-Eval scores.')
    parser.add_argument(
        '--coherence_path',
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        '--consistency_path',
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        '--fluency_path',
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        '--relevance_path',
        type=str,
        required=False,
        default=None,
    )

    args = parser.parse_args()

    return args


def parse_output(
    output
) -> float:
    """
    Parses the output string to extract a numerical score.

    Args:
        output (str): The output string containing a numerical score.

    Returns:
        float: The extracted score as a float. Returns 0 if no score is found
        or if an error occurs during conversion.
    """
    matched = re.search(r"^ ?([\d\.]+)", output)
    if (matched):
        try:
            score = float(matched.group(1))
        except Exception:
            score = 0
    else:
        score = 0
    return score


def parse_geval_outputs(
    output: List[Dict[str, Any]],
) -> List[float]:
    """
    Parses all the G-Eval outputs to extract numerical scores.
    """
    scores = []
    for line in output:
        for res in line['all_responses']:
            scores.append(parse_output(res))

    return scores


def load_jsonl_file(
    filepath: str
) -> List[Dict[str, Any]]:
    """Load a json into a list
    *arguments*
    *filepath* path to the file
    """
    data = []
    with open(filepath, "r", encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line.strip()))
    return data


def load_json_file(
    filepath: str
) -> List[Any]:
    """Load a json into a list
    *arguments*
    *filepath* path to the file
    """
    with open(filepath, 'r', encoding='utf8') as reader:
        json_data = json.load(reader)
    return json_data


if __name__ == "__main__":
    args = parse_arguments()

    outputs = {}
    if args.coherence_path is not None:
        outputs["coherence"] = load_json_file(args.coherence_path)
    if args.consistency_path is not None:
        outputs["consistency"] = load_json_file(args.consistency_path)
    if args.fluency_path is not None:
        outputs["fluency"] = load_json_file(args.fluency_path)
    if args.relevance_path is not None:
        outputs["relevance"] = load_json_file(args.relevance_path)

    all_scores = {}
    scores = {}

    for key, item in outputs.items():
        all_scores[key] = parse_geval_outputs(item)
        if not all_scores[key]:
            scores[key] = [.0, None]
        else:
            scores[key] = (sum(all_scores[key]) / len(all_scores[key]), np.std(all_scores[key]))

    print("{:<15} | {:<6}, {:<6}".format("metric", "value", "score"))
    print("---------------------------------")
    for key, item in scores.items():
        print("{:<15} | {:<6}, {:<6}".format(key, round(item[0], 2), round(item[1], 2)))
