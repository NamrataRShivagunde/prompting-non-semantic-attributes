from dataclasses import dataclass
import  datasets as datasets
import argparse
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import csv
from transformers import DataCollatorWithPadding
from tqdm import tqdm
import numpy as np
import openai
import subprocess
import json


def get_arguments():
    """Set the arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "datasetname", 
        help="dataset name e.g. rte",
        type=str,
        )

    parser.add_argument(
        "templatename", 
        default="rte_base",
        help="template name e.g. rte_base",
        type=str,
        )

 
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu"
        )

    parser.add_argument(
        "--num_shots",
        default=4,
        help="number of shots",
        type=int,
        )

    parser.add_argument(
        "--batch_size",
        default=4,
        help="batch_size",
        type=int,
        )
    
    parser.add_argument(
        "--seed",
        default=1,
        help="seed",
        type=int,
        )
    
    # parser.add_argument(
    #     "--key",
    #     default=0,
    #     help="openai key for prompting gpt3",
    #     type=str,
    #     )
    # # parser.add_argument(
    #     "--random",
    #     action='store_true', # flag if examples must be chosen randomy or not
    #     help="Boolean value suggesting if the in-context exampl should be chosen randomlly or not, True if random",
    #     )

    args = parser.parse_args()
    return args


@dataclass
class NLI():
    def __init__(self, temp):
        ''' temp (dict) : it contains all column from the template csv file
        '''
        self.targets = temp['targets'] # "yes;no"
        self.task = temp['task'] # "rte"
        self.query_template = temp['query_template']

        if self.task == 'rte':
            LM_targets = self.targets.split(';')
            self.class_id_to_label = {
                    0: LM_targets[0],  # True
                    1: LM_targets[1]}  # False
    
    def apply_template(self, example, template):
        ''' get the example and apply template on it. 
            example = {premise:..., hypothesis:..., label: 0}
            template =  "Premise:{premise} Hypothesis:{hypothesis} label:{label}"
        '''
        premise = example['premise'] 
        hypothesis = example['hypothesis']
        label = example['label']

        # change label id to its word equivalent
        label_word = self.class_id_to_label[int(label)]
     
        example_filled = template.replace('{premise}', premise)
        example_filled = example_filled.replace('{hypothesis}', hypothesis) # filled template

        if '{label}' in example_filled: # for demostrations only
            example_filled = example_filled.replace('{label}', label_word)

        return example_filled

    def label_mapping(self):
        ''' maps the label word to label id
        '''
        if self.task == 'rte':
            LM_targets = self.targets.split(';')
            self.class_id_to_label = {
                    0: LM_targets[0],  # True
                    1: LM_targets[1]}  # False
        
        return self.class_id_to_label
    
    def process_example(self, example):
        ''' take a query and apply template to it
        '''
       
        premise = example['premise']
        hypothesis = example['hypothesis']
        label_id = example['label']
        label_word = (self.label_mapping())[int(label_id)]
        
        
        example = {'premise':premise,   # query
                    'hypothesis': hypothesis,
                    'label': label_id}

        filled_example =   self.apply_template(example, self.query_template) # single filled query
        
        return filled_example, label_word


def main():
    
    # get arguments
    args = get_arguments()

    # get dataset
    train_set = datasets.load_dataset('super_glue', args.datasetname, split='train') # to get few shot in-context examples
    dev_set = datasets.load_dataset('super_glue', args.datasetname, split='validation') # to evaluate 

    # get template
    temp = {}
    with open('templates.csv') as p_file:
        reader = csv.DictReader(p_file)
        for row in reader:
            if row['task'] == args.datasetname:
                if row['template_name'] == args.templatename:
                    temp['task'] = row['task']
                    temp['templatename'] = row['template_name']
                    temp['instruction'] = row['instruction']
                    temp['demo_template'] = row['template-demo']
                    temp['query_template'] = row['template-query'] # this is same as demo_tempalte without the label placeholder
                    temp['targets'] = row['targets'] # label names
        
    # initialize class
    data_cat = NLI(temp)

    prompt = ''
    if args.num_shots > 0:
        # create prompt (instructions + in-context exmaples with templates)
        # choose random n integers
        seed=args.seed
        random.seed(seed)
        random_ints =  random.sample(range(0, len(train_set)), args.num_shots) # from train_set choose n demos randomly
        
        few_shots = []
        # apply template to demostrations and add it to few_shots list
        for num in random_ints:
            filled_example = data_cat.apply_template(train_set[num],  temp['demo_template'])
            few_shots.append(filled_example)  
        
        if temp['instruction'] != '':
            prompt = temp['instruction'] + "\n" + "\n".join(few_shots) + "\n" # Before
        else:
            prompt = "\n".join(few_shots) + "\n"
    
    if args.num_shots == 0:
        prompt = temp['instruction']

    all_predictions = []
    all_true_labels = []
    target_words =  ['True','False']
    
    # evaluation loop 
    with torch.no_grad():
        for i in tqdm(range(len(dev_set))):
            example = dev_set[i]
           
            filled_example, label_word = data_cat.process_example(example)
            if prompt != '':
                filled_example = prompt + "\n" + filled_example # add instrcution if it exists

            completion_params = {
                        "engine": "text-davinci-003",
                        "prompt": filled_example,
                        "temperature": 1,
                        "logprobs": 1,
                        "max_tokens": 1,
                        "n": 1,
                        }

            gpt3result = openai.Completion.create(
                **completion_params,
            )

            label = gpt3result["choices"][0]["text"]
            print(label)
            label = label.split("\n")[0].strip().lower().capitalize()
            if label not in target_words:
                label = "Unknown"
            
            all_predictions.append(label)
            
            all_true_labels.append(label_word)
            
          
        print(all_predictions)
        accuracy =  (np.array(all_predictions) == np.array(all_true_labels)).mean()
        print("Accuracy for ", args.datasetname,", ", args.templatename, ", ", accuracy)


if __name__=='__main__':
        main()