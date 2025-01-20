import argparse
from openai import AsyncOpenAI
import asyncio
from typing import List, Dict, Any
import glob
import os
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run OpenAI Inference on a given doc')
    parser.add_argument(
        '--input_prompts',
        type=str,
        required=True,
        help='The input file in JSON format.',
    )
    parser.add_argument(
        '--output_prompts',
        type=str,
        required=True,
        help='The input file in JSON format.',
    )
    parser.add_argument(
        '--document_type',
        type=str,
        required=True,
        help='The document type.',
    )
    parser.add_argument(
        '--model',
        type=str,
        required=False,
        help='The model used to execute  the prompt generation.',
        default="gpt4",
    )
    parser.add_argument(
        '--openai_key',
        type=str,
        required=True,
        help='The openai key.',
    )

    args = parser.parse_args()

    return args


def make_dir(
    path: str
) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_txt_file(
    filepath: str,
    join: bool = False,
) -> List[str]:
    """Load a json into a list
    *arguments*
    *filepath* path to the file
    """
    data = []
    with open(filepath, 'r', encoding='utf8') as reader:
        lines = reader.readlines()
        for line in lines:
            data.append(line)
    if join:
        data = "".join(data)
    return data


def write_txt_file(
    filepath: str,
    input_list: List[str],
    mode: str = 'a+',
    overwrite: bool = False
) -> None:
    """Write a list into a txt
    *arguments*
    *filepath* path to save the file into
    *input_list* list to be saved in the json file, must be made of strings
    *overwrite* whether to force overwriting a file.
        When set to False you will append the new items to an existing jsonl file (if the file already exists).
    """
    if overwrite:
        try:
            os.remove(filepath)
        except Exception:
            pass
    with open(filepath, mode, encoding='utf8') as writer:
        for line in input_list:
            writer.write(line + '\n')


async def generate_openai(
    messages: List[Dict[str, Any]],
    client: AsyncOpenAI,
    model: str,
) -> str:
    """
    Runs a G-Eval prompt using OpenAI's API.
    """
    _response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
    )
    return _response.choices[0].message.content


PROMPT_END = {}

PROMPT_END['coh_detailed'] = """Source Text:

{{Document}}

Summary:

{{Summary}}


Evaluation Form (scores ONLY):

- Coherence:"""
PROMPT_END['con_detailed'] = """Source Text:

{{Document}}

Summary:

{{Summary}}


Evaluation Form (scores ONLY):

- Consistency:"""
PROMPT_END['flu_detailed'] = """Summary:

{{Summary}}


Evaluation Form (scores ONLY):

- Fluency (1-3):"""
PROMPT_END['rel_detailed'] = """Source Text:

{{Document}}

Summary:

{{Summary}}


Evaluation Form (scores ONLY):

- Relevance:"""


async def main():
    args = parse_arguments()

    make_dir(args.output_prompts)

    client = AsyncOpenAI(api_key=args.openai_key, max_retries=10)
    model = args.model

    tasks = []
    prompt_paths = glob.glob(f"{args.input_prompts}/*.txt")
    prompt_paths = list(filter(lambda x: 'flu_detailed.txt' not in x, prompt_paths))
    for path in prompt_paths:
        prompt = load_txt_file(path, join=True)
        tasks.append(generate_openai(
            messages=[{"role": "system", "content": prompt.format(**{
                "document_type": args.document_type
            })}],
            client=client,
            model=model,
        ))
    outputs = await asyncio.gather(*tasks)
    for path, output in zip(prompt_paths, outputs):
        prompt_name = os.path.basename(path).replace(".txt", "")
        prompt = load_txt_file(path, join=True).format(**{
            'document_type': args.document_type
        })
        prompt += f"\n{output.strip()}\n\n\n{PROMPT_END[prompt_name]}"
        write_txt_file(
            f"{args.output_prompts}/{prompt_name}.txt",
            [prompt],
            overwrite=True,
        )
    prompt = load_txt_file(f"{args.input_prompts}/flu_detailed.txt", join=True).format(**{
        'document_type': args.document_type
    })
    prompt += f"\n\n\n{PROMPT_END['flu_detailed']}"
    write_txt_file(
        f"{args.output_prompts}/flu_detailed.txt",
        [prompt],
        overwrite=True,
    )


if __name__ == '__main__':
    asyncio.run(main())
