import re
import requests
import json
import torch
import transformers
from transformers import BitsAndBytesConfig

class Model():
    def __init__(self, config, cur_stage, cur_round=0):
        self.dataset_name = config.dataset_name
        self.config = config

        self.repeat = 10

        self.refresh_stage(cur_stage, cur_round)
        
    def refresh_stage(self, cur_stage, cur_round):
        self.cur_stage = cur_stage
        self.cur_round = cur_round    
        assert cur_stage in ['Prepare-Model', 'COT-Initial-Generation', 'Contrast-Responses-Merge-Memory', 'Initial-Regeneration', 'Regeneration-w-Suggestion']

        if cur_stage in ['Contrast-Responses-Merge-Memory', 'Regeneration-w-Suggestion']:
            self.cur_exp_name = f'{cur_round}-{cur_stage}'
        else:
            self.cur_exp_name = cur_stage
    
    # Only for Llama
    def prepare_model(self, ckpt):
        # LLAMA Specific config
        self.device = torch.device("cuda")
        self.pipeline = transformers.pipeline(
        "text-generation",
        model=ckpt,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto"
        )
        
        self.terminators = [
        self.pipeline.tokenizer.eos_token_id,
        self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def _generate(self, templated_example):
        if 'llama' in self.config.model_name.lower():
            completion = self._llama_generate(instruction=templated_example)
        elif 'gpt' in self.config.model_name.lower():
            completion = self._gpt35_api(prompt=templated_example)
        elif 'qwen' in self.config.model_name.lower():
            completion = self._qwen_turbo_api(prompt=templated_example)
        else:
            print('unknown model', flush=True)
        return completion
            

    def _select_template(self, example_dict: dict):
        cur_stage = self.cur_stage
        cur_round = self.cur_round

        if 'latest-memory' not in example_dict or 'N/A' in example_dict['latest-memory']:
            if cur_stage == 'Contrast-Responses-Merge-Memory':
                template_path = f"prompt_template/{self.config.dataset_name}/{cur_stage}/template_wo_memory.txt"
            elif cur_stage == 'Regeneration-w-Suggestion':
                cur_stage = 'Initial-Regeneration'
                template_path = f"prompt_template/{self.config.dataset_name}/{cur_stage}/template.txt"
            else:
                template_path = f"prompt_template/{self.config.dataset_name}/{cur_stage}/template.txt"
        else:
            if cur_stage == 'Contrast-Responses-Merge-Memory':
                template_path = f"prompt_template/{self.config.dataset_name}/{cur_stage}/template_w_memory.txt"
            else:
                template_path = f"prompt_template/{self.config.dataset_name}/{cur_stage}/template.txt"

        with open(template_path, 'r', encoding='utf-8') as fr:
            self.template = fr.read()
            


    def predict(self, example_dict: dict):
        self._select_template(example_dict)

        templated_example = self._apply_template(template=self.template, example=example_dict)

        completion = self._generate(templated_example)

        if self.cur_stage in ['COT-Initial-Generation', 'Initial-Regeneration', 'Regeneration-w-Suggestion']:
            answer = self.derive_answer_from_completion(example=example_dict, completion=completion)
            for _ in range(self.repeat):
                if isinstance(answer, bool):
                    break
                if 'invalid' not in answer:
                    break
                completion = self._generate(templated_example)
                answer = self.derive_answer_from_completion(example=example_dict, completion=completion)
            output = {f"{self.cur_exp_name}-answer": answer,
                    f"{self.cur_exp_name}": completion}
        elif self.cur_stage in ['Contrast-Responses-Merge-Memory']:
            if self.search_stop_sign_from_completion(completion=completion):
                return {self.cur_exp_name: 'NO DIFFERENCE'}
            if 'latest-memory' not in example_dict:
                suggestion = self.derive_suggestion_from_completion(completion=completion)
                for _ in range(self.repeat):
                    if 'N/A' not in suggestion:
                        break
                    completion = self._generate(templated_example)
                    # print(completion, flush=True)
                    if self.search_stop_sign_from_completion(completion=completion):
                        return {self.cur_exp_name: 'NO DIFFERENCE'}
                    suggestion = self.derive_suggestion_from_completion(completion=completion)
                output = {self.cur_exp_name+'-compared-key': example_dict['cur-selected-key'] ,self.cur_exp_name: suggestion, self.cur_exp_name+'-completion': completion, 'latest-memory': suggestion}
            else:
                checklist = self.derive_checklist_from_completion(completion=completion)
                for _ in range(self.repeat):
                    if 'N/A' not in checklist:
                        break
                    completion = self._generate(templated_example)
                    if self.search_stop_sign_from_completion(completion=completion):
                        return {self.cur_exp_name: 'NO DIFFERENCE'}
                    checklist = self.derive_checklist_from_completion(completion=completion)
                output = {self.cur_exp_name+'-compared-key': example_dict['cur-selected-key'] ,self.cur_exp_name: checklist, self.cur_exp_name+'-completion': completion, 'latest-memory': checklist}

        return output

    def _apply_template(self, template: str, example: dict):
        # for every [{FIELD}] in the template, replace it with the corresponding value of the key "{field}" in the example dict
        example_in_template = template
        cur_round = self.cur_round
        cur_stage = self.cur_stage

        for field in re.findall(r"\[.*?\]", template):
            field_name = field[1:-1]
            field_name = field_name.lower()

            # tracking the latest E-step or M-step
            if cur_stage == 'Contrast-Responses-Merge-Memory':
                if field_name == 'candidate':
                    if cur_round == 1:
                        field_name = 'Initial-Regeneration'
                    else:
                        field_name = f'{cur_round-1}-Regeneration-w-Suggestion'
                elif field_name == 'memory':
                    field_name = 'latest-memory'
            elif cur_stage == 'Regeneration-w-Suggestion' and field_name == 'suggestion':
                # print(latest_memory, flush=True)
                field_name = 'latest-memory'
            
            assert field_name in example
            example_in_template = example_in_template.replace(field, str(example[field_name]))
        return example_in_template

    def _qwen_turbo_api(self, prompt):
        # Please replace this part with your own api interface
        return 
    
    def _gpt35_api(self, prompt):
        # Please replace this part with your own api interface
        return

    def _llama_generate(self, instruction):
        try:
            messages = [
                {"role": "user", "content": instruction},
            ]
            
            prompt = self.pipeline.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
            )

            outputs = self.pipeline(
            prompt,
            max_new_tokens=1024,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=0.4,
            pad_token_id = self.pipeline.tokenizer.eos_token_id,
            use_cache=True
        )
            return outputs[0]["generated_text"][len(prompt):]
        except Exception as e:
            print(e)
            return 'Error'

    def derive_answer_from_completion(self, example, completion):
        answer = self._extract(example=example, completion=completion)
        answer = self.postprocess_answer(answer)
        return answer

    def derive_suggestion_from_completion(self, completion):
        match = re.search(r'<SUGGESTION>(.*)', completion, re.DOTALL)
        if match:
            result = match.group(1)
            suggestion = result.strip(': is')
        else:
            # print(f"No matched suggestion found in {completion}", flush=True)
            suggestion = 'N/A'
        return suggestion

    def derive_checklist_from_completion(self, completion):
        match = re.search(r'<CHECKING>(.*)', completion, re.DOTALL)
        if match:
            result = match.group(1)
            suggestion = result.strip(': is')
        else:
            # print(f"No matched suggestion found in {completion}", flush=True)
            suggestion = 'N/A'
        return suggestion

    def search_stop_sign_from_completion(self, completion):
        match = re.search(r'<STOP!>', completion, re.DOTALL)
        return match

    def _extract(self, example: dict, completion: str):
        if self.dataset_name in ["GSM8K", "SVAMP", "MultiArith", "ASDiv"]:
            if "answer is " not in completion:
                answer = "[invalid]"
            else:
                answer = completion.split("answer is ")[-1].strip("\n.")
        elif self.dataset_name == "StrategyQA":
            if "answer is" not in completion:
                answer = "[invalid]"
            else:
                answer = completion.split("answer is ")[-1].split()[0].strip("\n.").lower()
                if answer == "yes":
                    answer =  True
                elif answer == "no":
                    answer = False
                else:
                    answer = "[invalid]"
        elif self.dataset_name == "Date":
            if "answer is " not in completion:
                answer = "[invalid]"
            else:
                answer = completion.split("answer is ")[-1].strip()
                answer = re.sub(pattern="[\s\.#]", repl="", string=answer)
        else:
            print('Unknown dataset for answer extraction!')
        return answer

    def postprocess_answer(self, answer):
        if self.dataset_name in ["GSM8K", "SVAMP", "MultiArith", "ASDiv"]:
            answer = str(answer).strip()
            answer = answer.split("\n")[-1]
        elif self.dataset_name == "Date":
            answer = str(answer).strip()
            answer = answer.split("\n")[-1]
            answer = answer.rstrip("Y")
        elif self.dataset_name == "StrategyQA":
            pass
        else:
            print('Unknown dataset for post-processing!')
        return answer