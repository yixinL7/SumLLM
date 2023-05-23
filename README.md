# SumLLM
This repo contains the training and evaluation scripts, model outputs, and LLM-based evaluation results for our paper: On Learning to Summarize with Large Language Models as References.

## How to install


## Description of Codes
- `config.py` -> model configuration
- `data_utils.py` -> dataloader
- `main.py` -> training script of contrastive learning (BRIO)
- `main_mle.py` -> training script of MLE
- `model.py` -> models and loss functions
- `utils.py` -> utility functions
- `gen_candidate.py` -> generate candidate summaries

### Workspace
The directory `./cache` should be created for our experiments, which will store all the experiment results and model checkpoints.

## How to run
In this section, we will introduce how to run the code for our experiments.
Please note that while our script supports multi-gpu training, all the configurations are based on single-gpu training so you may need to modify the configurations (e.g. per-gpu batch size) for multi-gpu training.

For the MLE training, any GPU with 16GB memory should be sufficient. For the contrastive learning, at least 24GB memory is required.


### Warmup Training
At the first step, we need to train the model with MLE for warmup using the ChatGPT-generated summaries.
Please run the following command:
```
python main_mle.py --cuda --gpuid [list of gpuid] --config chatgpt_warmup -l 
```
The checkpoint based on the validation MLE loss (`cache/*/model_mle`) will be used for the next step.

### Experiment 1: Learning with GPTScore

#### MLE Training
For the MLE training, please run the following command:
```
python main_mle.py --cuda --gpuid [list of gpuid] --config gpt3_mle --model_pt [path to the checkpoint from the warmup training] -l 
```
The checkpoint based on the validation MLE loss (`cache/*/model_mle`) will be used for evaluation.

We also need to train another checkpoint with less training data for the next step:
```
python main_mle.py --cuda --gpuid [list of gpuid] --config gpt3_mle_small --model_pt [path to the checkpoint from the warmup training] -l
```


#### Contrastive Learning
For the contrastive learning, please run the following command:
```
python main.py --cuda --gpuid [list of gpuid] --config gpt3_brio --model_pt [path to the checkpoint from the MLE training] -l
```
The checkpoint based on the validation contrastive loss (`cache/*/model_ranking`) will be used for evaluation.

### Experiment 2: Learning with GPTRank using ChatGPT

#### MLE Training
For the MLE training, please run the following command:
```
python main_mle.py --cuda --gpuid [list of gpuid] --config chatgpt_mle --model_pt [path to the checkpoint from the warmup training] -l 
```
This is similar to the MLE training in the warmup step, but we use more training data here.
The checkpoint based on the validation MLE loss (`cache/*/model_mle`) will be used for evaluation.

#### Contrastive Learning
For the contrastive learning, please run the following command:
```
python main.py --cuda --gpuid [list of gpuid] --config chatgpt_brio --model_pt [path to the checkpoint from the warmup training] -l
```
The checkpoint based on the validation contrastive loss (`cache/*/model_ranking`) will be used for evaluation.

### Experiment 3: Learning with GPTRank using GPT-4

#### MLE Training
For the MLE training, please run the following command:
```
python main_mle.py --cuda --gpuid [list of gpuid] --config gpt4_mle --model_pt [path to the checkpoint from the warmup training] -l 
```
The checkpoint based on the validation MLE loss (`cache/*/model_mle`) will be used for evaluation.

#### Contrastive Learning
For the contrastive learning, please run the following command (we will use the checkpoint from the warmup training here):
```
python main.py --cuda --gpuid [list of gpuid] --config gpt4_brio --model_pt [path to the checkpoint from the warmup training] -l
```
The checkpoint based on the validation contrastive loss (`cache/*/model_ranking`) will be used for evaluation.


