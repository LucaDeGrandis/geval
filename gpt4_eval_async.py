import json
import argparse
import tqdm
from openai import AsyncOpenAI
import asyncio
from typing import List, Dict, Any


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run OpenAI Inference on a given doc')
    parser.add_argument(
        '--prompt_fp',
        type=str,
        required=True,
        default="prompts/summeval/con_detailed.txt",
    )
    parser.add_argument(
        '--save_fp',
        type=str,
        required=True,
        default="results/gpt4_con_detailed_openai.json",
    )
    parser.add_argument(
        '--summeval_fp',
        type=str,
        required=True,
        default="data/summeval.json",
    )
    parser.add_argument(
        '--key',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
    )

    args = parser.parse_args()

    return args


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
        temperature=2,
        max_tokens=5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        n=20,
    )
    all_responses = [choice.message.content for choice in _response.choices]
    return all_responses


async def main():
    args = parse_arguments()

    summeval = json.load(open(args.summeval_fp))
    prompt = open(args.prompt_fp).read()

    client = AsyncOpenAI(api_key=args.key, max_retries=10)
    model = args.model

    # Generate with OpenAI's API.
    tasks = []
    for instance in tqdm.tqdm(summeval):
        source = instance['source']
        system_output = instance['system_output']
        cur_prompt = prompt.replace('{{Document}}', source).replace('{{Summary}}', system_output)
        instance['prompt'] = cur_prompt
        tasks.append(generate_openai(
            messages=[{"role": "system", "content": cur_prompt}],
            client=client,
            model=model,
        ))
    outputs = await asyncio.gather(*tasks)

    new_json = []
    for instance, output in zip(summeval, outputs):
        instance['all_responses'] = output
        new_json.append(instance)

    with open(args.save_fp, 'w') as f:
        json.dump(new_json, f, indent=4)


if __name__ == '__main__':
    asyncio.run(main())
