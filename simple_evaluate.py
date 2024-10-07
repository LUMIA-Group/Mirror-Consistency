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
				print(pred_answer, flush=True)
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
	print(f'correct_count: {correct_count}, total_count: {total_count}')
	acc=round(correct_count/ total_count * 100, 1)
	return acc