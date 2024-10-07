import os
import sys
import argparse

from pprint import pprint
from tqdm import tqdm

from copy import deepcopy

from utils import *
from model import Model



def single_run(llm, stage, recording, config, round):
    # Initialization of LLM Wrapper
    llm.refresh_stage(cur_stage = stage, cur_round = round)
    
    # Current experiment name
    if stage in ['Contrast-Responses-Merge-Memory', 'Regeneration-w-Suggestion']:
        exp_name = f'{round}-{stage}'
    else:
        exp_name = stage
        
    for sample in tqdm(recording):
        if exp_name in sample.keys():
            print(f'{exp_name} already done for the {sample["id"]}-th sample')
            continue

        try:
            completion = llm.predict(sample)
            for k,v in completion.items():
                sample[k] = v
        except Exception as e:
            sample[exp_name] = str(e)
            print(f'Error at {sample["id"]}-th sample: {str(e)}', file=sys.stderr)

    # Save current recording-List
    recording_path = f'/ossfs/workspace/Faithful-COT-Logic/recording/{config.model_name}/{config.dataset_name}'
    if not os.path.exists(recording_path):
        os.makedirs(recording_path)
    with open(os.path.join(recording_path, f'{exp_name}-{config.start_index}-{config.end_index}.json'), 'w') as f:
        json.dump(recording, f, indent=4)

def complete_run(llm, recording, config, total_iter_rounds):
    try:
        single_run(llm=llm, stage='Initial-Regeneration', recording=recording, config=config, round=0)
        get_cur_major_vote(weight_method='average', recording=recording, config=config)
        for iter_step_id in range(1, total_iter_rounds+1):
            single_run(llm=llm, stage='Contrast-Responses-Merge-Memory', recording=recording, config=config, round=2*iter_step_id-1)
            single_run(llm=llm, stage='Regeneration-w-Suggestion', recording=recording, config=config, round=2*iter_step_id)
            get_cur_major_vote(weight_method='average', recording=recording, config=config)
    except Exception as e:
        print(f'Error: {str(e)}', file=sys.stderr)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mirror-Consistency Experiment Runner')

    # Configuration parameters
    parser.add_argument('--dataset_name', type=str, default='GSM8K', help='Name of the dataset to use')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use (e.g., train, test)')
    parser.add_argument('--model_name', type=str, default='LLAMA3-8B', help='Name of the model to use')
    parser.add_argument('--model_path', type=str, default='/Meta-Llama-3-8B-Instruct', help='Path to the model weights (only for LLAMA models)')
    parser.add_argument('--start_index', type=int, default=0, help='Start index for data slicing')
    parser.add_argument('--end_index', type=int, default=768, help='End index for data slicing')
    parser.add_argument('--total_iter_rounds', type=int, default=9, help='Total iteration rounds for complete_run')

    # Parse arguments
    args = parser.parse_args()

    # Dataset configuration
    dataset_frn = f"data/{args.dataset_name}/{args.split}.jsonl"
    dataset = load_data(dataset_frn)

    print(f'Dataset: {args.dataset_name}, Length: {len(dataset)}')

    # Path of the Initial Responses
    initial_pred_directory = f'Initial-Generation-List/{args.model_name}/{args.dataset_name}'
    initial_pred_path = os.path.join(initial_pred_directory, 'output.jsonl')

    # Read and Process the Initial Responses
    initial_generation_list = read_jsonl_as_list(initial_pred_path)
    recording_list = [
        {
            'id': data_item['id'],
            'question': data_item['question'],
            'response': gen_item['completion'],
            'response-answer': gen_item['answer']
        }
        for data_item, gen_item in zip(dataset, initial_generation_list)
    ]

    # Use recording_list to keep track of all the intermediate results.
    # Now the keys in recording_list: 'id', 'question', 'response' (initial response)
    print(f'Size of initial prediction: {len(recording_list)}')

    # Model Initialization
    llm = Model(args, cur_stage='Prepare-Model')

    if "llama" in args.model_name.lower():
        llm.prepare_model(args.model_path)

    # Running
    recording = deepcopy(recording_list)[args.start_index:args.end_index]
    print(f'Size of current run: {len(recording)}')

    complete_run(llm, recording, args, total_iter_rounds=args.total_iter_rounds)