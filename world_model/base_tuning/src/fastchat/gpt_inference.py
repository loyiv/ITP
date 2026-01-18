from openai import OpenAI
import os
from tqdm import tqdm
import time
import json
import multiprocessing
import argparse
from conversation import get_conv_template

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str)
    parser.add_argument('--output_dir', type=str, default="results")
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--api_key_file', type=str, default='openai_api_key.txt')
    parser.add_argument('--base_url', type=str, default="https://api.openai.com/v1/")
    parser.add_argument('--num_proc', type=int, default=1)
    parser.add_argument('--max_tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--top_p', type=float, default=0.75)

    return parser.parse_args()

def load_api(path: str, is_pool: bool = False):
    api_keys = []
    with open(path, 'r') as f:
        for line in f:
            key = line.strip()
            api_keys.append(key)
    if is_pool:
        return api_keys
    else:
        return api_keys[0]

def get_uniq_id(sample):
    return '_'.join([sample['dialog_id'], sample['turn_id']])

def load_finished_data(log_file):
    finished_ids = []
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                js = json.loads(line)
                uid = get_uniq_id(js)
                finished_ids.append(uid)
    return finished_ids

def get_prompt(data_point):
    conv = get_conv_template(name="vicuna_v1.1")

    if "instruction" in data_point and str(data_point["instruction"]).strip() != "":
        conv.set_system_message(data_point["instruction"])

    conversations = data_point["conversations"].copy()
    assert conversations[-1]["from"] == "gpt"
    conversations[-1]["value"] = ""

    for utt in conversations:
        if utt["from"] == "human":
            conv.append_message(conv.roles[0], utt["value"])
        else:
            conv.append_message(conv.roles[1], utt["value"])

    prompt = conv.get_prompt()

    return prompt

def gpt_completion(item: tuple):
    idx, args, prompt_block, api_key, output_path = item

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    if os.path.exists(output_path):
        finished_ids = load_finished_data(output_path)
        output_f = open(output_path, 'a')
    else:
        finished_ids = []
        output_f = open(output_path, 'w')
    print(f'Worker {idx} start', 'total:', len(prompt_block), 'finished:', len(finished_ids))

    for sample in tqdm(prompt_block, total=len(prompt_block), desc=f'Worker {idx}'):
        task_id = get_uniq_id(sample)
        if task_id in finished_ids:
            continue

        flag = False
        prompt = sample['prompt']
        while not flag:
            try:
                completion = client.chat.completions.create(
                    model=args.model,
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                )
                response = completion.choices[0].message.content
                sample['generated_response'] = response.strip()
                flag = True
            except Exception as e:
                print(f'Worker {idx}', e)
                time.sleep(5)

        del sample['prompt']
        output_f.write(json.dumps(sample) + '\n')
        output_f.flush()

if __name__ == "__main__":
    args = parse_args()
    print(args)

    api_key = load_api(args.api_key_file)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    test_data_prompts = []
    with open(args.test_data_path, 'r') as f:
        test_data = json.load(f)
        for data_point in test_data:
            gold_response = data_point['conversations'][-1]['value']
            prompt = get_prompt(data_point)
            test_data_prompts.append(
                {
                    "dialog_id": data_point['dialog_id'],
                    "turn_id": data_point['turn_id'],
                    "prompt": prompt,
                    "gold_response": gold_response
                }
            )

    task_block = []
    l = len(test_data_prompts) // args.num_proc
    for i in range(args.num_proc):
        if i == args.num_proc - 1:
            prompt_block = test_data_prompts[i*l:]
        else:
            prompt_block = test_data_prompts[i*l:(i+1)*l]
        output_path = os.path.join(args.output_dir, '{}_output_block{}.jsonl'.format(
            args.test_data_path.split('/')[-1].split('.')[0], i))
        task_block.append((i, args, prompt_block, api_key, output_path))

    pool = multiprocessing.Pool(args.num_proc)
    pool.map(gpt_completion, task_block)

    output_file = os.path.join(args.output_dir, args.test_data_path.split('/')[-1].split('.')[0] + '_output.jsonl')
    with open(output_file, 'w') as f:
        for i in range(args.num_proc):
            block_file = os.path.join(args.output_dir, '{}_output_block{}.jsonl'.format(
                args.test_data_path.split('/')[-1].split('.')[0], i))
            with open(block_file, 'r') as bf:
                for line in bf:
                    f.write(line)
            os.remove(block_file)
    print(f'Inference on test set finished, results saved in {output_file}')
