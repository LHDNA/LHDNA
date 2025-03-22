import os
from datasets import load_dataset
from transformers import BartModel
from transformers import LlamaForCausalLM, AutoTokenizer,BartTokenizer,BartModel,OPTForCausalLM,Trainer, TrainingArguments
import functools
import json
import pandas as pd
import datasets
from datasets import Dataset

@functools.lru_cache(1)
def read_all_contexts():
    dataset = datasets.load_from_disk("dataset/coqa")
    return {_['id']: _['story'] for _ in dataset}

def _preprocess():
    save_path="dataset/coqa"
    if not os.path.exists(save_path):
        with open('data/coqa/coqa-dev-v1.0.json', 'r') as infile:
            data = json.load(infile)['data']
        dataset = {}
        dataset['story'] = []
        dataset['question'] = []
        dataset['answer'] = []
        dataset['additional_answers'] = []
        dataset['id'] = []

        for sample_id, sample in enumerate(data):
            story = sample['story']
            questions = sample['questions']
            answers = sample['answers']
            additional_answers = sample['additional_answers']
            for question_index, question in enumerate(questions):
                dataset['story'].append(story)
                dataset['question'].append(question['input_text'])
                dataset['answer'].append({
                    'text': answers[question_index]['input_text'],
                    'answer_start': answers[question_index]['span_start']
                })
                dataset['id'].append(sample['id'] + '_' + str(question_index))
                additional_answers_list = []

                for i in range(3):
                    additional_answers_list.append(additional_answers[str(i)][question_index]['input_text'])

                dataset['additional_answers'].append(additional_answers_list)
                story = story + ' Q: ' + question['input_text'] + ' A: ' + answers[question_index]['input_text']
                if not story[-1] == '.':
                    story = story + '.'

        dataset_df = pd.DataFrame.from_dict(dataset)

        dataset = Dataset.from_pandas(dataset_df)
        dataset.save_to_disk(save_path)
    return save_path

def get_dataset(tokenizer, split='validation'):
    dataset = datasets.load_from_disk(_preprocess())
    id_to_question_mapping = dict(zip(dataset['id'], dataset['question']))

    def encode_coqa(example):
        example['answer'] = example['answer']['text']
        example['prompt'] = prompt = example['story'] + ' Q: ' + example['question'] + ' A:'
        return tokenizer(prompt, truncation=False, padding=False)


    dataset = dataset.map(encode_coqa, batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)

    return dataset

def _generate_config(tokenizer):

    if tokenizer.__class__.__name__ == 'LlamaTokenizer':
        eos_token_id = [tokenizer.encode(_)[-1] for _ in ['.', '\n']] + [29889]  # seems to be '.' as well
        #eos_token_id = [tokenizer(_)['input_ids'] for _ in ['\n', ',', '.']]
    elif tokenizer.__class__.__name__ == 'GPT2Tokenizer':
        try:
            eos_token_id = [tokenizer.encode(_)[1] for _ in ['.', '\n']]
        except:
            eos_token_id = [tokenizer.encode(_)[0] for _ in ['.', '\n']]
    elif tokenizer.__class__.__name__ == "PreTrainedTokenizerFast":
        eos_token_id = [tokenizer(_)['input_ids'][-1] for _ in ['\n', ',', '.']]
    else:
        eos_token_id = [tokenizer(_)['input_ids'][-1] for _ in ['\n', ',', '.']]
    eos_token_id += [tokenizer.eos_token_id]
    question_framing_ids = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:']
    # Follows Kuhn et al 2023 as Llama does not have CoQA
    try:
        question_framing_ids = [[tokenizer(eos_token)['input_ids'][1]] for eos_token in question_framing_ids]
    except:
        question_framing_ids = [[tokenizer(eos_token)['input_ids'][0]] for eos_token in question_framing_ids]
    return dict(eos_token_id=eos_token_id, bad_words_ids=question_framing_ids)


if __name__=="__main__":
    pass