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
        "modelname",
        help="huggingface modelname e.g. facebook/opt-125m",
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

    args = parser.parse_args()
    return args


@dataclass
class NLI():
    def __init__(self, temp):
        ''' temp (dict) : it contains all column from the template csv file
        '''
        self.targets = temp['targets'] # "True;False"
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

        filled_example =   self.apply_template(example, self.query_template)  # single filled query
        
        return filled_example, label_word


def main():
    
    # get arguments
    args = get_arguments()

    # load tokenizer and model
    modelname = args.modelname

    device_map = {
    'model.decoder.embed_tokens': 0,
    'model.decoder.embed_positions': 0,
    'model.decoder.final_layer_norm': 0,
    'model.decoder.layers.0': 0,
    'model.decoder.layers.1': 0,
    'model.decoder.layers.2': 0,
    'model.decoder.layers.3': 0,
    'model.decoder.layers.4': 0,
    'model.decoder.layers.5': 0,
    'model.decoder.layers.6': 0,
    'model.decoder.layers.7': 0,
    'model.decoder.layers.8': 0,
    'model.decoder.layers.9': 0,
    'model.decoder.layers.10': 0,
    'model.decoder.layers.11': 0,
    'model.decoder.layers.12': 0,
    'model.decoder.layers.13': 0,
    'model.decoder.layers.14': 0,
    'model.decoder.layers.15': 0,
    'model.decoder.layers.16': 0,
    'model.decoder.layers.17': 0,
    'model.decoder.layers.18': 0,
    'model.decoder.layers.19': 0,
    'model.decoder.layers.20': 0,
    'model.decoder.layers.21': 0,
    'model.decoder.layers.22': 0,
    'model.decoder.layers.23': 0,
    'model.decoder.layers.24': 1,
    'model.decoder.layers.25': 1,
    'model.decoder.layers.26': 1,
    'model.decoder.layers.27': 1,
    'model.decoder.layers.28': 1,
    'model.decoder.layers.29': 1,
    'model.decoder.layers.30': 1,
    'model.decoder.layers.31': 1,
    'model.decoder.layers.32': 1,
    'model.decoder.layers.33': 1,
    'model.decoder.layers.34': 1,
    'model.decoder.layers.35': 1,
    'model.decoder.layers.36': 1,
    'model.decoder.layers.37': 1,
    'model.decoder.layers.38': 1,
    'model.decoder.layers.39': 1,
    'model.decoder.layers.40': 1,
    'model.decoder.layers.41': 1,
    'model.decoder.layers.42': 1,
    'model.decoder.layers.43': 1,
    'model.decoder.layers.44': 1,
    'model.decoder.layers.45': 1,
    'model.decoder.layers.46': 1,
    'model.decoder.layers.47': 1,
    'lm_head': 1,
    }

    #load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(modelname,  device_map=device_map, load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(modelname, return_tensors="pt")

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
            # prompt = temp['instruction'] + "\n" + "\n".join(few_shots) + "\n" # Before
            prompt =  "\n".join(few_shots) + "\n" + temp['instruction'] + "\n" # After 
        else:
            prompt = "\n".join(few_shots) + "\n"
    
    if args.num_shots == 0:
        prompt = temp['instruction']
    
    all_predictions = []
    all_true_labels = []
    all_nextword = []
    target_ids = []

    target_words = temp['targets'].split(';') # e.g. ['True', 'False']
    target_encoded = tokenizer(target_words)  # e.g. {'input_ids': [[2, 36948], [2, 46659]], 'attention_mask': [[1, 1], [1, 1]]}
    for i in range(len(target_words)):
        target_ids.append(target_encoded['input_ids'][i][1])  # e.g. [36948, 46659]

    # evaluation loop 
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(dev_set))):
            example = dev_set[i]
           
            filled_example, label_word = data_cat.process_example(example)
            if prompt != '':
                filled_example = prompt + "\n" + filled_example # add instrcution if it exists
                # e.g. filled_example = Context:Dana Reeve, the widow of the actor Christopher Reeve, has died of l
                # lung cancer at age 44, according to the Christopher Reeve Foundation.
                # \nQuestion:Christopher Reeve had an accident.True or False?
            # print("------------",filled_example,"-------------")
            tok_input = tokenizer(filled_example, padding=True, return_tensors="pt")
            inputs = tok_input['input_ids'].to("cuda")
            output = model(inputs)
            
            # gather and compare logits of labels
            logits = output.logits[:,-1,:].squeeze().cpu() # [1, s, v] --> [1,v] --->[v]
            indices = torch.tensor(target_ids) # [len(target_ids)]

            choice_id = torch.gather(logits, 0, indices) # [len(target_id)]
            choice_id = choice_id.argmax(dim=0) # [1]

            # next word prediction
            nextword_id = logits.argmax(dim=0)
            nextword = tokenizer.decode(nextword_id)

            all_predictions.append(target_words[choice_id]) 
            all_true_labels.append(label_word)
            all_nextword.append(nextword.strip())


        accuracy =  (np.array(all_predictions) == np.array(all_true_labels)).mean()
        print("Accuracy for ", args.datasetname,", ", args.templatename, ", ", accuracy)



if __name__=='__main__':
        main()