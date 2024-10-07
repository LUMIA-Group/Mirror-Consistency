import os
from argparse import Namespace
from pprint import pprint
import numpy as np

from collections import defaultdict
from copy import deepcopy

import matplotlib.pyplot as plt

from utils import *



def evaluate_acc(dataset, predictions, dataset_name, non_empty_only=False, valid_only=False, key4check = 'Initial-Regeneration-answer-extracted'):
	correct_count, total_count = 0, 0
        
	for example, prediction in zip(dataset, predictions):
		gold_id = int(example["id"])
		if prediction == {}:
			continue
		pred_id = int(prediction["id"])

		try:
			assert gold_id == pred_id
		except:
			raise AssertionError(f"Gold id {gold_id} doesn't match pred id {pred_id}.")

		try:
			gold_answer = extract_gold_answer(dataset_name, example["answer"])
		except SyntaxError as e:
			print("Error: ", e)
			print(gold_id)
			exit(-1)
            
		if key4check not in prediction:
			continue
		pred_answer = extract_pred_answer(dataset_name, prediction[key4check])

		if non_empty_only and pred_answer == "":
			continue

		if valid_only:
			if type(pred_answer) == str and ("invalid" in pred_answer or "error" in pred_answer):
				# print(pred_answer, flush=True)
				continue

		total_count += 1
		try:
			correct = is_correct(dataset_name, gold_answer, pred_answer)
		except Exception as e:
			print("Error: ", e)
			print("Example: ", gold_id)
			print("Question: ", example["question"])
			print("Gold answer: ", gold_answer, type(gold_answer))
			print("Pred answer: ", pred_answer, type(pred_answer))
			print("Completion: ", prediction["completion"])
			print("\n")
			exit(-1)

		if correct:
			correct_count += 1      

		prediction[key4check+'-correct']=correct
	# print(f'correct_count: {correct_count}, total_count: {total_count}')
	acc=round(correct_count/ total_count * 100, 1)
	return acc

def get_gold_answer_list(dataset):
    gold_answer_list=[]
    for example in dataset:
        gold_id = int(example["id"])
        try:
            gold_answer = extract_gold_answer(config.dataset_name, example["answer"])
        except SyntaxError as e:
            print("Error: ", e)
            print(gold_id)
            exit(-1)
        gold_answer_list.append({'id':gold_id, 'answer':gold_answer})
    return gold_answer_list

from collections import defaultdict

def find_max_key(d):
    max_key = None
    max_value = float('-inf')
    for key, value in d.items():
        if value > max_value:
            max_key = key
            max_value = value
    return max_key



def complete_consistency_evaluation_pipeline(recording, seperate_statistics=True):
    recording_backup = deepcopy(recording)

    max_num = 0
    for k,v in recording_backup[0].items():
        if k.endswith('-answer'):
            if 'invalid' in str(v):
                continue
            rnd = k.split('-')[0]
            if rnd not in ['response', 'Initial']:
                max_num = max(max_num, int(rnd))
    print(f'num: {max_num}')

    num = max_num
    
    steps_name_list = ['response-answer'] + [f'{i}-Regeneration-w-Suggestion-answer' for i in range(num+1)]

    acc_list = []
    for step_name in steps_name_list:
        acc = evaluate_acc(dataset=dataset,
                           predictions=recording_backup,
                           dataset_name=config.dataset_name,
                           non_empty_only=True,
                           valid_only=True,
                           key4check=step_name)
        acc_list.append(acc)

    print(acc_list)
    print(f'avg: {np.mean(acc_list)}')
    
    gold_answer_list = get_gold_answer_list(dataset)
    
    for sample in recording_backup:
        for k,v in sample.copy().items():
            if k.endswith('answer'):
                pred_answer = extract_pred_answer(config.dataset_name, v)
    
                # if isinstance(pred_answer, str):
                #     print(pred_answer)
                sample[f'{k}-extracted'] = pred_answer

    # pprint(recording_backup[0])

    for weight_method in ['average','linear','exp']:

        selected_ans_list = []
        
        for r in recording_backup:
            ans = defaultdict(int)
            for k,v in r.items():
                if k.endswith('-extracted'):
                    if 'invalid' in str(v):
                        continue
                    rnd = k.split('-')[0]
                    if rnd in ['response', 'Initial']:
                        weight=1
                    else:
                        if int(rnd)>=num//2:
                            continue
                        weight = int(rnd)//2
                    if weight_method =='linear':
                        ans[v]+=weight
                    if weight_method =='exp':
                        ans[v]+=10**weight
                    if weight_method =='average':
                        ans[v]+=1
            max_ans = find_max_key(ans)
            selected_ans_list.append({'id':r['id'], 'ans':ans, 'selected-ans':max_ans})

        calibration_sum, correct_cnt, total_cnt, bin_ece_list, ece, conf_list = get_calibration(selected_ans_list, gold_answer_list)
                
        print(f'half-result: weight-method: {weight_method}, correct: {correct_cnt}, total: {total_cnt}, acc: {correct_cnt/total_cnt}, Brier score: {calibration_sum}, ECE: {ece}')


    
    for weight_method in ['average','linear','exp']:

        selected_ans_list = []
        
        for r in recording_backup:
            ans = defaultdict(int)
            for k,v in r.items():
                if k.endswith('-extracted'):
                    if 'invalid' in str(v):
                        continue
                    rnd = k.split('-')[0]
                    if rnd in ['response', 'Initial']:
                        weight=1
                    else:
                        weight = int(rnd)//2
                    if weight_method =='linear':
                        ans[v]+=weight
                    if weight_method =='exp':
                        ans[v]+=2**weight
                    if weight_method =='average':
                        ans[v]+=1
            max_ans = find_max_key(ans)
            selected_ans_list.append({'id':r['id'], 'ans':ans, 'selected-ans':max_ans})

        total_cnt=0
        correct_cnt=0
        
        for p in zip(selected_ans_list,gold_answer_list):
            total_cnt+=1
            if p[0]['selected-ans']==p[1]['answer']:
                correct_cnt+=1
                
        calibration_sum, correct_cnt, total_cnt, bin_ece_list, ece, conf_list = get_calibration(selected_ans_list, gold_answer_list)
                
        print(f'weight-method: {weight_method}, correct: {correct_cnt}, total: {total_cnt}, acc: {correct_cnt/total_cnt}, Brier Score: {calibration_sum}, ECE: {ece}')



def get_bin_ece_consistency(recording):
    recording_backup = deepcopy(recording)

    max_num = 0
    for k,v in recording_backup[0].items():
        if k.endswith('-answer'):
            if 'invalid' in str(v):
                continue
            rnd = k.split('-')[0]
            if rnd not in ['response', 'Initial']:
                max_num = max(max_num, int(rnd))
    # print(f'num: {max_num}')

    num = max_num
    
    steps_name_list = ['response-answer'] + [f'{i}-Regeneration-w-Suggestion-answer' for i in range(num+1)]

    acc_list = []
    for step_name in steps_name_list:
        acc = evaluate_acc(dataset=dataset,
                           predictions=recording_backup,
                           dataset_name=config.dataset_name,
                           non_empty_only=True,
                           valid_only=True,
                           key4check=step_name)
        acc_list.append(acc)

    # print(acc_list)
    # print(f'avg: {np.mean(acc_list)}')
    
    gold_answer_list = get_gold_answer_list(dataset)
    
    for sample in recording_backup:
        for k,v in sample.copy().items():
            if k.endswith('answer'):
                pred_answer = extract_pred_answer(config.dataset_name, v)
    
                # if isinstance(pred_answer, str):
                #     print(pred_answer)
                sample[f'{k}-extracted'] = pred_answer


    weight_method = 'average'

    selected_ans_list = []
    
    for r in recording_backup:
        ans = defaultdict(int)
        for k,v in r.items():
            if k.endswith('-extracted'):
                if 'invalid' in str(v):
                    continue
                rnd = k.split('-')[0]
                if rnd in ['response', 'Initial']:
                    weight=1
                else:
                    weight = int(rnd)//2
                if weight_method =='linear':
                    ans[v]+=weight
                if weight_method =='exp':
                    ans[v]+=2**weight
                if weight_method =='average':
                    ans[v]+=1
        max_ans = find_max_key(ans)
        selected_ans_list.append({'id':r['id'], 'ans':ans, 'selected-ans':max_ans})

    total_cnt=0
    correct_cnt=0
    
    for p in zip(selected_ans_list,gold_answer_list):
        total_cnt+=1
        if p[0]['selected-ans']==p[1]['answer']:
            correct_cnt+=1
            
    calibration_sum, correct_cnt, total_cnt, bin_ece_list, ece, conf_list = get_calibration(selected_ans_list, gold_answer_list)
            
    # print(f'weight-method: {weight_method}, correct: {correct_cnt}, total: {total_cnt}, acc: {correct_cnt/total_cnt}, Brier Score: {calibration_sum}, ECE: {ece}')

    return bin_ece_list, conf_list



def complete_evaluation_pipeline(recording, seperate_statistics=True):
    recording_backup = deepcopy(recording)

    steps_name_list = ['response-answer', 'Initial-Regeneration-answer']+[f'{rnd}-Regeneration-w-Suggestion-answer' for rnd in [2,4,6,8,10,12,14,16,18]]

    acc_list = []
    for step_name in steps_name_list:
        acc = evaluate_acc(dataset=dataset,
                           predictions=recording_backup,
                           dataset_name=config.dataset_name,
                           non_empty_only=True,
                           valid_only=True,
                           key4check=step_name)
        acc_list.append(acc)

    print(acc_list)
    print(f'avg: {np.mean(acc_list)}')
    
    gold_answer_list = get_gold_answer_list(dataset)
    
    for sample in recording_backup:
        for k,v in sample.copy().items():
            if k.endswith('answer'):
                pred_answer = extract_pred_answer(config.dataset_name, v)

                sample[f'{k}-extracted'] = pred_answer

    
    for weight_method in ['average','linear','exp']:

        selected_ans_list = []
        
        for r in recording_backup:
            ans = defaultdict(int)
            for k,v in r.items():
                if k.endswith('-extracted'):
                    if 'invalid' in str(v):
                        continue
                    rnd = k.split('-')[0]
                    if rnd in ['response', 'Initial']:
                        weight=1
                    else:
                        weight = int(rnd)//2
                    if weight_method =='linear':
                        ans[v]+=weight
                    if weight_method =='exp':
                        ans[v]+=2**weight
                    if weight_method =='average':
                        ans[v]+=1
            max_ans = find_max_key(ans)
            selected_ans_list.append({'id':r['id'], 'ans':ans, 'selected-ans':max_ans})

        total_cnt=0
        correct_cnt=0
        
        for p in zip(selected_ans_list,gold_answer_list):
            total_cnt+=1
            if p[0]['selected-ans']==p[1]['answer']:
                correct_cnt+=1
                
        calibration_sum, correct_cnt, total_cnt, bin_ece_list, ece, conf_list = get_calibration(selected_ans_list, gold_answer_list)
                
        print(f'weight-method: {weight_method}, correct: {correct_cnt}, total: {total_cnt}, acc: {correct_cnt/total_cnt}, Brier Score: {calibration_sum}, ECE: {ece}')



def get_calibration(selected_ans_list, gold_answer_list):
    conf_score_list=[]
        
    max_conf=0
    min_conf=1
    
    
    for example in selected_ans_list:
        ans_dict = example['ans']
        selected = example['selected-ans']
        # print(list(ans_dict.values()))
        tot = np.sum(list(ans_dict.values()))
        if tot==0.0:
            continue
        conf = ans_dict[selected] / tot
        max_conf = max(max_conf, conf)
        min_conf = min(min_conf, conf)
        conf_score_list.append({'id':example['id'], 'conf_score': conf, 'selected-ans': selected})
    
    for example in conf_score_list:
        example['conf_score'] = (example['conf_score']-min_conf)/(max_conf-min_conf+1e-6)
    ###
    
    total_cnt=0
    correct_cnt=0
    
    conf_list=[]
    invalid_cnt = 0
    for i,p in enumerate(conf_score_list):
        total_cnt+=1
        while gold_answer_list[i+invalid_cnt]['id']!=p['id']:
            invalid_cnt+=1
        q=gold_answer_list[i+invalid_cnt]
        if p['selected-ans']==q['answer']:
            correct_cnt+=1
            conf_list.append({'id':p['id'], 'accuracy':1.0, 'conf':p['conf_score']})
        else:
            # print(f"right: {p[0]['selected-ans']}, correct: {p[1]['answer']}")
            conf_list.append({'id':p['id'], 'accuracy':0.0, 'conf':p['conf_score']})
    
    calibration_sum=0
    for example in conf_list:
        calibration_sum+=(example['accuracy']-example['conf'])**2
    calibration_sum/=len(conf_list)
    
    total_cnt-=invalid_cnt

    ### ECE
    accuracy_part = np.array([_['accuracy'] for _ in conf_list])
    conf_part = np.array([_['conf'] for _ in conf_list])
    
    n_bins = 10
    bin_limits = np.linspace(0, 1, n_bins+1)
    ece=0

    bin_ece_list = []
    for i in range(n_bins):
        bin_mask = (conf_part>=bin_limits[i]) & (conf_part<bin_limits[i+1])
        bin_accuracy_part = accuracy_part[bin_mask]
        bin_conf_part = conf_part[bin_mask]
    
        if len(bin_accuracy_part) == 0:
            bin_ece_list.append([0., 0.])
            continue
    
        bin_accuracy = np.mean(bin_accuracy_part)
        bin_conf = np.mean(bin_conf_part)
    
        bin_ece = np.abs(bin_accuracy-bin_conf)*len(bin_accuracy_part)/len(accuracy_part)
        bin_ece_list.append([bin_conf, bin_accuracy])
        
        ece+=bin_ece

    return calibration_sum, correct_cnt, total_cnt, bin_ece_list, ece, conf_list




def get_bin_ece_mirror(recording):
    recording_backup = deepcopy(recording)

    steps_name_list = ['response-answer', 'Initial-Regeneration-answer']+[f'{rnd}-Regeneration-w-Suggestion-answer' for rnd in [2,4,6,8,10,12,14,16,18]]

    acc_list = []
    for step_name in steps_name_list:
        acc = evaluate_acc(dataset=dataset,
                           predictions=recording_backup,
                           dataset_name=config.dataset_name,
                           non_empty_only=True,
                           valid_only=True,
                           key4check=step_name)
        acc_list.append(acc)

    # print(acc_list)
    # print(f'avg: {np.mean(acc_list)}')
    
    gold_answer_list = get_gold_answer_list(dataset)
    
    for sample in recording_backup:
        for k,v in sample.copy().items():
            if k.endswith('answer'):
                pred_answer = extract_pred_answer(config.dataset_name, v)
    
                # if isinstance(pred_answer, str):
                #     print(pred_answer)
                sample[f'{k}-extracted'] = pred_answer


    weight_method = 'average'

    selected_ans_list = []
    
    for r in recording_backup:
        ans = defaultdict(int)
        for k,v in r.items():
            if k.endswith('-extracted'):
                if 'invalid' in str(v):
                    continue
                rnd = k.split('-')[0]
                if rnd in ['response', 'Initial']:
                    weight=1
                else:
                    weight = int(rnd)//2
                if weight_method =='linear':
                    ans[v]+=weight
                if weight_method =='exp':
                    ans[v]+=2**weight
                if weight_method =='average':
                    ans[v]+=1
        max_ans = find_max_key(ans)
        selected_ans_list.append({'id':r['id'], 'ans':ans, 'selected-ans':max_ans})

    total_cnt=0
    correct_cnt=0
    
    for p in zip(selected_ans_list,gold_answer_list):
        total_cnt+=1
        if p[0]['selected-ans']==p[1]['answer']:
            correct_cnt+=1
            
    calibration_sum, correct_cnt, total_cnt, bin_ece_list, ece, conf_list = get_calibration(selected_ans_list, gold_answer_list)
            
    # print(f'weight-method: {weight_method}, correct: {correct_cnt}, total: {total_cnt}, acc: {correct_cnt/total_cnt}, Brier Score: {calibration_sum}, ECE: {ece}')

    return bin_ece_list, conf_list






if __name__ == '__main__':
    config = Namespace()

    dataset_name_list = ['Date', 'GSM8K', 'SVAMP', 'StrategyQA']
    model_name_list = ['LLAMA3-8B']

    recording_path = '/ossfs/workspace/Faithful-COT-Logic/recording'
    for dataset_name in dataset_name_list:
        for model_name in model_name_list:
            dir_path = f'{recording_path}/{model_name}/{dataset_name}'
            if os.path.exists(dir_path):
                recordings = os.listdir(dir_path)
                if 'final.json' in recordings:
                    print(f'model_name: {model_name}, dataset_name: {dataset_name}, final')
                    with open(f'{dir_path}/final.json', 'r') as f:
                        recording = json.load(f)
                elif 'half-final.json' in recordings:
                    print(f'model_name: {model_name}, dataset_name: {dataset_name}, half-final')
                    with open(f'{dir_path}/half-final.json', 'r') as f:
                        recording = json.load(f)
                else:
                    continue
                config.dataset_name = dataset_name
                config.split = 'test'
                config.model_name = model_name
                
                dataset_frn = f"data/{config.dataset_name}/{config.split}.jsonl"
                dataset = load_data(dataset_frn)

                complete_evaluation_pipeline(recording, seperate_statistics=False)
                
                
                bin_ece_list, conf_list = get_bin_ece_mirror(recording)
                acc_part = np.array(bin_ece_list)[:,1]
                bin_limits = np.linspace(0, 1, 11)
                non_zero_mask = acc_part!=0.0
                acc_part = acc_part[non_zero_mask]
                bin_limits=(bin_limits[1:]+bin_limits[:-1])/2
                bin_limits = bin_limits[non_zero_mask]

                plt.plot(bin_limits, acc_part, alpha=0.5, marker='o', label='Mirror-Consistency')
                with open(f'{dataset_name}-{model_name}.txt', 'w') as f:
                    f.write('Mirror Bin Limits: ' + str(bin_limits) + '\n')
                    f.write('Mirror Acc Part: ' + str(acc_part) + '\n')
                

            dir_path = f'{recording_path}/{model_name}/{dataset_name}/consistency_baseline/'
            if os.path.exists(dir_path):
                recordings = os.listdir(dir_path)
                if 'final.json' in recordings:
                    print(f'model_name: {model_name}, dataset_name: {dataset_name}, final')
                    with open(f'{dir_path}/final.json', 'r') as f:
                        recording = json.load(f)
                elif 'half-final.json' in recordings:
                    print(f'model_name: {model_name}, dataset_name: {dataset_name}, half-final')
                    with open(f'{dir_path}/half-final.json', 'r') as f:
                        recording = json.load(f)
                else:
                    continue
                config.dataset_name = dataset_name
                config.split = 'test'
                config.model_name = model_name
                
                dataset_frn = f"data/{config.dataset_name}/{config.split}.jsonl"
                dataset = load_data(dataset_frn)

                complete_consistency_evaluation_pipeline(recording, seperate_statistics=False)


                bin_ece_list, conf_list = get_bin_ece_consistency(recording)
                acc_part = np.array(bin_ece_list)[:,1]
                bin_limits = np.linspace(0, 1, 11)
                non_zero_mask = acc_part!=0.0
                acc_part = acc_part[non_zero_mask]
                bin_limits=(bin_limits[1:]+bin_limits[:-1])/2
                bin_limits = bin_limits[non_zero_mask]
                plt.plot(bin_limits, acc_part, alpha=0.5, marker='o', label='Self-Consistency')
                plt.plot([0,1],[0,1],'k--', label='Perfect Calibration')
                plt.legend(fontsize=13)
                plt.xlabel('Bins of confidence level', fontsize=14)
                plt.ylabel('Accuracy', fontsize=14)
                plt.savefig(f'{dataset_name}-{model_name}.pdf')
                with open(f'{dataset_name}-{model_name}.txt', 'a') as f:
                    f.write('Consistency Bin Limits: ' + str(bin_limits) + '\n')
                    f.write('Consistency Acc Part: ' + str(acc_part) + '\n')
                plt.show()
                
                print()