import json
import jsonlines
import re
import csv
from collections import Counter, defaultdict
from fractions import Fraction
import math
import copy
import random


INVALID_ANS = "[invalid]"

NO_CODE_STOP_TOKEN = {"GSM8K": "Q:",
                      "SVAMP": "Q:",
                      "MultiArith": "Q:",
                      "ASDiv": "Q:",
                      "AQUA": "Q: ",
                      "StrategyQA": "Q:",
                      "Date": "Q:",
                      "sports": "Q:",
                      "saycan": "Human:",
                      "CLUTRR": "Context:",
                      }

def load_data(frn):
	'''Load data from a file.
	:param frn (str): The dataset file name.

	:return: The dataset (a list of examples, each as a dictionary).
	'''
	if frn.endswith(".jsonl"):
		with open(frn, 'r') as fr:
			lines = []
			for i, line in enumerate(fr):
				if line.strip() == "":
					continue
				try:
					lines.append(json.loads(line))
				except json.decoder.JSONDecodeError as e:
					print(f"Error in line {i}: {line}\n {e}")
					exit(-1)
		return lines
	elif frn.endswith(".csv"):
		with open(frn) as fr:
			reader = csv.DictReader(fr)
			return [line for line in reader]



def str2num(answer_str, rounding="int", abs_val=True):
	'''Convert a string to a number.
	@:param answer_str (str): The string to convert.
	@:param rounding (str): The rounding method for the answer. Can be "int", "ceil", or "floor".
	@:param abs_val (bool): Whether to take the absolute value of the answer.

	@:return The converted number.
	'''
	if "/" in answer_str:
		answer_str =  float(sum(Fraction(s) for s in answer_str.split()))
	answer_str = float(answer_str)

	if rounding == "int":
		answer_str = int(answer_str)
	elif rounding == "ceil":
		answer_str = math.ceil(answer_str)
	elif rounding == "floor":
		answer_str = math.floor(answer_str)

	if abs_val:
		answer_str = abs(answer_str)

	return answer_str


def extract_gold_answer(dataset_name, gold_completion):
	'''Extract the gold answer from a completion.
	:param dataset_name (str): The name of the dataset.
	:param gold_completion (str): The gold completion.

	:return: The gold answer.
	'''
	if dataset_name in ["GSM8K", "SVAMP", "MultiArith"]:
		ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
		match = ANS_RE.search(gold_completion)
		if match:
			match_str = match.group(1).strip()
			match_str = match_str.replace(",", "")
			return int(match_str)
		else:
			return INVALID_ANS
	elif dataset_name == "ASDiv":
		# ASDiv has questions with multi-value answers, e.g., Q: "What the width and the height of xxx?", A: (5, 10)
		if type(gold_completion) in [tuple, list]: # the gold answer has multiple values
			answer = dict(Counter([int(ans) for ans in gold_completion]))
		else: # the gold answer has a single value
			answer = int(gold_completion)
		return answer
	elif dataset_name in ["Date", "CLUTRR"]:
		answer = gold_completion.split("#### ")[-1]
		return answer
	elif dataset_name == "saycan":
		answer = eval(gold_completion)
		return answer
	elif dataset_name in ["StrategyQA"]:
		answer = bool(gold_completion)
		return answer
	elif dataset_name in ["sports"]:
		answer = bool(int(gold_completion))
		return answer
	else:
		return gold_completion

def extract_pred_answer(dataset_name, pred_completion, rounding="int", abs_val=True):
	'''Extract the predicted answer from a completion.
	:param dataset_name (str): The name of the dataset.
	:param pred_completion (str): The predicted completion.
	:param rounding (str): The rounding method for the predicted answer. Can be "int", "ceil", or "floor".
	:param abs_val (bool): Whether to take the absolute value of the predicted answer.

	:return: The predicted answer.
	'''
	if INVALID_ANS in str(pred_completion):
		return INVALID_ANS

	if dataset_name in ["GSM8K", "SVAMP", "MultiArith"]:
		# GSM8K, SVAMP, and MultiArith all have a single-value integer answer
		if type(pred_completion) == int:
			pred_answer = pred_completion
		elif type(pred_completion) == str:
			ANS_RE = re.compile(r"(\-?[0-9\.\,]+)")
			match = ANS_RE.search(pred_completion)
			if match:
				match_str = match.group(1).strip()
				match_str = match_str.replace(",", "")
				try:
					pred_answer = str2num(match_str, rounding, abs_val)
				except:
					pred_answer = INVALID_ANS
			else:
				pred_answer = INVALID_ANS
		return pred_answer

	elif dataset_name in ["ASDiv"]:
		# ASDiv has questions with multi-value answers, e.g., Q: "What the width and the height of xxx?", A: (5, 10)
		if type(pred_completion) == int:
			return pred_completion
		elif type(pred_completion) == str:
			pred_completion = pred_completion.lstrip("{([").rstrip("]})")
			pred_answers = pred_completion.split(",")
			final_pred_answers = []
			for pred_answer in pred_answers:
				pred_answer = pred_answer.strip().split(":")[-1].strip("'\"")
				try:
					pred_answer = str2num(pred_answer, rounding, abs_val)
					final_pred_answers.append(pred_answer)
				except ValueError:
					continue
			if len(final_pred_answers) > 1:
				return dict(Counter(final_pred_answers))
			elif len(final_pred_answers) == 1:
				return final_pred_answers[0]
			else:
				return INVALID_ANS
		elif type(pred_completion) == dict:
			new_dict = {}
			for key, value in pred_completion.items():
				new_key = str(key)
				new_key = str2num(new_key, rounding, abs_val)
				new_dict[new_key] = value
			return new_dict

	elif dataset_name in ["StrategyQA"]:
		answer = bool(pred_completion)
		return answer

	elif dataset_name in ["sports"]:
		answer = bool(int(pred_completion))
		return answer

	elif dataset_name in ["saycan"]:
		answer = pred_completion.strip()
		return answer

	else:
		return pred_completion



def is_correct(dataset_name, gold_answers, pred_answer):
	'''Check if a predicted answer is correct.
	:param dataset_name (str): The name of the dataset.
	:param gold_answers: The gold answer(s).
	:param pred_answer: The predicted answer.

	:return: Whether the prediction is correct (True) or not (False).
	'''

	# saycan has multiple correct plans, so we need to check if the predicted plan is in the list of correct plans
	if dataset_name == "saycan":
		assert type(gold_answers) == list
		assert type(pred_answer) == str
		if pred_answer in ["[error]", "[invalid]"]:
			return False
		else:
			pred_answer = pred_answer.replace("\\n", "\n")
			pred_plan_list = []
			step_count = 0
			steps = re.split(r", |\n", pred_answer.strip())
			for step in steps:
				step_cols = step.split(". ")
				if len(step_cols) != 2:
					return "[invalid]"
				step_action = step_cols[1]
				if "find(initial)" in step_action:
					continue
				step_count += 1
				new_step = f"{step_count}. {step_action}"
				pred_plan_list.append(new_step)
			for gold_answer in gold_answers:
				gold_plan_list = gold_answer.strip().split("\n")
				if pred_plan_list == gold_plan_list:
					return True
		return False

	else:	# all other datasets have a single correct answer
		gold_answer = gold_answers
		return pred_answer == gold_answer



def read_jsonl_as_list(json_path):
    data_list = []
    with open(json_path, 'r') as f:
        reader = jsonlines.Reader(f)
        for line in reader:
            data_list.append(line)
    return data_list

def dump_list_as_jsonl(json_path, data_list):
    with open(json_path, 'w') as f:
        writer = jsonlines.Writer(f)
        for line in data_list:
            writer.write(line)



def get_cur_major_vote(weight_method, recording, config):
    selected_ans_list = []
    recording_backup = copy.deepcopy(recording)

    def find_max_key(d):
        max_key = None
        max_value = float('-inf')
        for key, value in d.items():
            if value > max_value:
                max_key = key
                max_value = value
        return max_key

    for sample in recording_backup:
        for k,v in sample.copy().items():
            if k.endswith('answer'):
                pred_answer = extract_pred_answer(config.dataset_name, v)
    
                # if isinstance(pred_answer, str):
                #     print(sample)
                sample[f'{k}-extracted'] = pred_answer
  
    for r in recording_backup:
        ans = defaultdict(int)
        for k,v in r.items():
            if k.endswith('-extracted'):
                if 'invalid' in str(v):
                    continue
                rnd = k.split('-')[0]
                if rnd in ['Initial', 'response']:
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

    for id, s in enumerate(selected_ans_list):
        selected_ans = s['selected-ans']
        major_key_choices=[]
        assert recording_backup[id]['id']==s['id']
        for k,v in recording_backup[id].items():
            if k.endswith('-answer-extracted'):
                if selected_ans==v:
                    major_key_choices.append(k.split('-answer-extracted')[0])
        assert len(major_key_choices)>0
        selected_key = random.choice(major_key_choices)
        assert recording[id]['id']==s['id']
        selected_completion = recording[id][selected_key]
        recording[id]['cur-selected-key']=selected_key
        recording[id]['cur-selected-response']=selected_completion