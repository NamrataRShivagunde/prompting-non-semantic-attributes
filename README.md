# prompting-non-semantic-attributes 

The code is to load SuperGlue datasets and evaluate them on OPT for specified template.
Currently the code supports 
Dataset - RTE
Model - OPT-30b and GPT-3

Install the libraries by using

        pip install -r requirements.txt
 


For opt run

        python main.py [DATASETNAME] [TEMPLATENAME] [MODELNAME] --seed [SEED] --num_shots [NUM OF SHOTS]
        e.g. python main.py rte tf-s facebook/opt-30b --seed 25 --num_shots 0

For GPT3 set the OpenAI key variable

        python gpt3.py [DATASETNAME] [TEMPLATENAME] [MODELNAME] --seed [SEED] --num_shots [NUM OF SHOTS] 
        e.g. python gpt3.py rte tf-s --seed 25 --num_shots 0 
