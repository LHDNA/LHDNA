import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import coqa
import torch
from transformers import AutoModelForCausalLM, LlamaForCausalLM, AutoTokenizer,BartTokenizer,BartModel,OPTForCausalLM,Trainer, TrainingArguments
import functools
import pandas as pd
import transformers
import numpy as np
import tqdm
import pandas as pd
from torch.utils.data import SubsetRandomSampler
import torch.nn as nn
from metric import *
import re
import warnings
import torch.nn.functional as F
import triviaqa
import nq_open
import squad
warnings.filterwarnings("ignore")

cuda =torch.cuda.is_available()

if cuda:
    device = torch.device('cuda')
else:
    device=torch.device('cpu')


NUM_GENERATIONS_PER_PROMPT=10
TOP_P=0.99
TOP_K=10
TEMPERATURE=0.5
HOOKS={}
ACTIVATIONS=[]
MODEL=""

def one_hot_batched(token_ids,vocab_size):
    batch_size, num_tokens = token_ids.shape
    return torch.zeros(batch_size, num_tokens, vocab_size, device=token_ids.device).scatter_(-1, token_ids.unsqueeze(-1), 1.)

def get_embeddings(model_embeddings,input_ids):
    embedding_matrix=model_embeddings
    vocab_size=embedding_matrix.shape[0]
    one_hot_tensor = one_hot_batched(input_ids, vocab_size).to(torch.float16).to(device)
    token_ids_tensor_one_hot = one_hot_tensor.clone().requires_grad_(True)
    inputs_embeds = torch.matmul(token_ids_tensor_one_hot, embedding_matrix)
    return inputs_embeds, token_ids_tensor_one_hot


# shape of input_[0]: (batch_size*sequence_length,neurons) or (batch_size,sequence_length,neurons)
# https://github.com/jalammar/ecco
def get_activations_hook(name, input_):
    global ACTIVATIONS
    global MODEL
    layer_number = re.search(r"(?<=\.)\d+(?=\.)", name).group(0)
    if "Llama" in MODEL or "Qwen" in MODEL:
            ACTIVATIONS.append(input_[0].detach().cpu().numpy())
    else:
        ACTIVATIONS.append(input_[0].detach().cpu().numpy())


def attach_hook(model, activations_layer):
    global HOOKS
    for name, module in model.named_modules():
        if re.search(activations_layer, name):
            # Adjusted lambda to handle all 3 arguments
            HOOKS[name] = module.register_forward_hook(
                lambda module, input_, output, name=name: get_activations_hook(name, input_)
            )

def remove_hooks():
    global HOOKS
    global ACTIVATIONS
    for handle in HOOKS.values():
        handle.remove()
    HOOKS={}
    ACTIVATIONS=[]

 
def seed_everything(seed: int):
    if seed is None:
        return
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataset_fn(data_name):
    if data_name == 'triviaqa':
        return triviaqa.get_dataset
    if data_name == 'coqa':
        return coqa.get_dataset
    if data_name == 'nq_open':
        return nq_open.get_dataset
    if data_name =="squad":
        return  squad.get_dataset
    
def get_num_tokens(generation):  # generation: num_seq x max(num_tokens)
    num_tokens = []
    for ids in generation:
        count = 0
        for id in ids:
            if id>2:
                count+=1
        # num_tokens.append(count)
        num_tokens.append(count+1)
    return num_tokens
    
def get_generation_config(input_ids, tokenizer, data_name):
    generation_config=None
    assert len(input_ids.shape) == 2
    max_length_of_generated_sequence = 256
    # max_length_of_generated_sequence = 50
    if data_name == 'triviaqa':
        generation_config = triviaqa._generate_config(tokenizer)
    if data_name == 'coqa':
        generation_config = coqa._generate_config(tokenizer)
    if data_name == 'nq_open':
        generation_config=nq_open._generate_config(tokenizer)
    if data_name == 'squad':
        generation_config=squad._generate_config(tokenizer)
    generation_config['max_new_tokens'] = max_length_of_generated_sequence
    generation_config['early_stopping'] = True
    # https://jaketae.github.io/study/gpt2/#setup
    generation_config['pad_token_id'] = tokenizer.eos_token_id
    return generation_config

@functools.lru_cache()
def load_model_and_tokenizer(model_name="opt-1.3b",device="cuda",torch_dtype=torch.float16):
    activation=""
    model_root="/nvme1n1/LLM/"+model_name
    if model_name.startswith('opt-'):
        model=OPTForCausalLM.from_pretrained(
            model_root,
            device_map="auto",
            torch_dtype=torch_dtype,
            )
        activation="fc2"
    elif model_name.startswith('Meta') or "Llama" in model_name or "Qwen" in model_name:
        model=AutoModelForCausalLM.from_pretrained(model_root,device_map="auto",torch_dtype=torch_dtype)
        activation="down_proj"
    elif model_name=="gpt2":
        model=AutoModelForCausalLM.from_pretrained(model_root,device_map="auto",torch_dtype=torch_dtype)
        activation="mlp.c_proj"

    if model_name.startswith('opt-'):
        tokenizer=AutoTokenizer.from_pretrained(model_root,use_fast=False)
    elif model_name.startswith('Meta') or "Llama" in model_name or "Qwen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_root, cache_dir=None, use_fast=False)
        tokenizer.eos_token_id = 2
        tokenizer.bos_token_id = 1
        tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)
        tokenizer.bos_token = tokenizer.decode(tokenizer.bos_token_id)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer=AutoTokenizer.from_pretrained(model_root,use_fast=False)
    return model,tokenizer,activation
    
                            

@torch.no_grad()
def get_generations(model_name,dataset_name,alpha=0.5,beta=0.5,seed=None,max_num_gen_once=10,do_sample=False):
    global MODEL
    MODEL=model_name
    model, tokenizer,activation=load_model_and_tokenizer(model_name,device)
    # embeddings_layer_name=""
    # if "opt" in model_name:
    #     embeddings_layer_name="model.decoder.embed_tokens.weight"
    # embed_retriever = attrgetter(embeddings_layer_name)
    # model_embeddings = embed_retriever(model)
    seed_everything(seed)
    dataset=get_dataset_fn(dataset_name)(tokenizer)
    if not do_sample:
        dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    else:
        num_samples = 2000
        sampler = SubsetRandomSampler(indices=np.random.choice(len(dataset), num_samples, replace=False))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=sampler)

    old_sequences=[]
    old_sequences={_['id']: _ for _ in old_sequences}
    sequences = []

    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        if batch['id'][0] in old_sequences:
            sequences.append(old_sequences[batch['id'][0]])
            continue
        input_ids = batch['input_ids'].to(device)
        input_length = input_ids.shape[1]
        generation_config=get_generation_config(input_ids,tokenizer,dataset_name)
        generation_config=transformers.GenerationConfig(**generation_config)
        # remove_hooks()
        # attach_hook(model,activation)
        layers=len(HOOKS)
        dict_outputs = model.generate(input_ids, attention_mask=batch['attention_mask'].to(device),
                                                            num_beams=1,
                                                            do_sample=False,
                                                            generation_config=generation_config,
                                                            output_hidden_states=True,
                                                            return_dict_in_generate=True,
                                                            output_scores=True,
                                                            min_new_tokens=1,
                                                            # output_attentions=True,
                                                            )
        most_likely_generations = dict_outputs.sequences.cpu()[0, input_length:]
        best_generated_text = tokenizer.decode(most_likely_generations, skip_special_tokens=True)
        # prediction_ids,prediction_scores=dict_outputs.sequences[0][input_length:], dict_outputs.scores
        # tmp=input_ids
        # pad_token_id = model.config.pad_token_id
        # eos_token_id = model.config.eos_token_id
        # pad_token_id=torch.tensor(pad_token_id).to(device)
        # eos_token_id=torch.tensor(eos_token_id).to(device)
        # attention_mask=batch['attention_mask'].to(device)
        # remove_hooks()
        # for pred_index, prediction_id in enumerate(prediction_ids):
        #     if pred_index == len(prediction_ids) - 1:
        #         encoder_input_embeds, _=get_embeddings(model_embeddings,tmp)
        #         decoder_input_embeds= None
        #         attach_hook(model,activation)
        #         extra_forward_kwargs = {
        #             'attention_mask': attention_mask, 
        #             'decoder_inputs_embeds': decoder_input_embeds}
        #         forward_kwargs = {
        #             'inputs_embeds': encoder_input_embeds,
        #             'use_cache': False,
        #             'return_dict': True,
        #             **{k: v for k, v in extra_forward_kwargs.items() if k in inspect.signature(model.forward).parameters}
        #         }
        #         _ = model(**forward_kwargs)
        #     tmp = torch.cat(
        #             [tmp, torch.tensor([[prediction_id]], device=device)],
        #             dim=-1
        #         )
        #     if getattr(model, '_prepare_attention_mask_for_generation'):
        #             assert len(tmp.size()) == 2 # will break otherwise
        #             attention_mask = model._prepare_attention_mask_for_generation(tmp, pad_token_id, eos_token_id)
        #             attention_mask = attention_mask.to(device)
        # most_likely_activations=[]
        # for index in range(layers,len(ACTIVATIONS),layers):
        #     most_likely_activations.append(ACTIVATIONS[index:index+layers])
        # if len(most_likely_activations)==0:
        #     print(batch_idx)
        #     print(len(ACTIVATIONS))
        #     print(tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True))
        #     print(best_generated_text)
        #     continue
        # most_likely_activations=np.asarray(most_likely_activations)#shape should be (num_tokens,num_layers,batch_size,num_neurons)
        # if len(most_likely_activations.shape)==5:
        #     most_likely_activations=most_likely_activations[:,:,:,0,:]

        # remove_hooks()
            
        attach_hook(model,activation)
        del dict_outputs
        torch.cuda.empty_cache()

        generations=[]
        num_gens=NUM_GENERATIONS_PER_PROMPT
        while num_gens>0:
            dict_outputs=model.generate(input_ids, attention_mask=batch['attention_mask'].to(device),
                                num_beams=1, num_return_sequences=min(max_num_gen_once, num_gens),
                                do_sample=True, top_p=TOP_P, top_k=TOP_K,
                                temperature=TEMPERATURE, 
                                generation_config=generation_config,
                                output_hidden_states=True,return_dict_in_generate=True,
                                output_scores=True,
                                # min_new_tokens=1
                                # output_attentions=True,
                                )
            generation = dict_outputs.sequences[:, input_length:].cpu()
            generations.append(generation)
            num_tokens = get_num_tokens(generation)
            num_gens -= len(generation)
            del dict_outputs

        torch.cuda.empty_cache()
        activations=[]
        for index in range(layers,len(ACTIVATIONS),layers):
            activations.append(ACTIVATIONS[index:index+layers])
        activations=np.asarray(activations)
        if len(activations.shape)==5:
            activations=activations[:,:,:,0,:]
        if len(activations) ==0 :
            continue
        generations = torch.nested.nested_tensor(generations).to_padded_tensor(tokenizer.eos_token_id)
        generations = generations.reshape(-1, generations.shape[-1])[:NUM_GENERATIONS_PER_PROMPT]
        generated_texts = [tokenizer.decode(_, skip_special_tokens=True) for _ in generations]
        # most_likely_activation_map=get_activation_map(most_likely_activations)
        activation_maps=get_activation_map(activations,num_tokens)
        lhdna=compute_matrix_similarities(activation_maps,alpha,beta)


        curr_seq = dict(
            prompt=tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True),
            id=batch['id'][0],
            question=batch['question'][0],
            answer=batch['answer'][0],
            additional_answers=[],
        )
        curr_seq.update(
            dict(
                most_likely_generation_ids = most_likely_generations,
                generations_ids=generations,
            )
        )
        curr_seq.update(
            dict(
                most_likely_generation=tokenizer.decode(curr_seq['most_likely_generation_ids'], skip_special_tokens=True),
                generations=generated_texts,
            )
        )
        curr_seq.update(
            dict(
                lhdna=lhdna
            )
        )
        if dataset_name == 'coqa' or dataset_name == 'TruthfulQA':
            curr_seq['additional_answers'] = [x[0] for x in batch['additional_answers']]

        sequences.append(curr_seq)
        # print(asizeof.asizeof(curr_seq))
        torch.cuda.empty_cache()
    return sequences
        

def main(model_name,dataset,sample=False,exp_id=0):
    print(f'Generating {NUM_GENERATIONS_PER_PROMPT} generations per prompt for {model_name} on {dataset}...')
    result_root="results/"+model_name+"_"+dataset
    sequences =get_generations(model_name,dataset,result_root=result_root,do_sample=sample)
    os.makedirs(result_root, exist_ok=True)
    exp_id=exp_id
    pd.to_pickle(sequences,os.path.join(result_root,f'{exp_id}.pkl'))

if __name__=="__main__":
    exp_id=0
    main("opt-1.3b",'coqa',False,exp_id)





    
