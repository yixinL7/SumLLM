# SumLLM
This repo contains the training and evaluation scripts, experiment data, model outputs for our paper: "[On Learning to Summarize with Large Language Models as References](https://arxiv.org/abs/2305.14239)".

This repo is intended and licensed for research use only. 
The data and model outputs are licensed under CC BY-NC 4.0 (allowing only non-commercial use).


## Quick Links

- [Requirements](#requirements)
- [Description of Codes](#description-of-codes)
  - [Workspace](#workspace)
- [Training](#training)
  - [Warmup Training](#warmup-training)
  - [Experiment 1: Learning with GPTScore](#experiment-1-learning-with-gptscore)
    - [MLE Training (BART.GPT3D3)](#mle-training-bartgpt3d3)
    - [Contrastive Learning (BRIO.GPT3D3)](#contrastive-learning-briogpt3d3)
  - [Experiment 2: Learning with GPTRank using ChatGPT](#experiment-2-learning-with-gptrank-using-chatgpt)
    - [MLE Training (BART.ChatGPT)](#mle-training-bartchatgpt)
    - [Contrastive Learning (BRIO.ChatGPT)](#contrastive-learning-briochatgpt)
  - [Experiment 3: Learning with GPTRank using GPT-4](#experiment-3-learning-with-gptrank-using-gpt-4)
    - [MLE Training (BART.GPT-4)](#mle-training-bartgpt-4)
    - [Contrastive Learning (BRIO.GPT-4)](#contrastive-learning-briogpt-4)
- [Evaluation](#evaluation)
  - [Output Generation](#output-generation)
  - [LLM-based Evaluations](#llm-based-evaluations)
    - [GPTScore](#gptscore)
    - [GPTRank](#gptrank)
- [Data](#data)
- [Model Outputs](#model-outputs)


## Requirements
We use Python 3.8, PyTorch 1.12.1 and Transformers 4.21.2 for our experiments. Please refer to `requirements.txt` for more requirements.

## Description of Codes
- `config.py` -> model configuration
- `data_utils.py` -> dataloader
- `main.py` -> training script of contrastive learning (BRIO)
- `main_mle.py` -> training script of MLE
- `model.py` -> models and loss functions
- `utils.py` -> utility functions
- `test.py` -> evaluation script
- `gpt_score.py` -> GPTScore evaluation script
- `gpt_rank.py` -> GPTRank evaluation script
- `gpt.py` -> OpenAI's GPT-3 API and other utility functions

### Workspace
The directory `./cache` should be created for our experiments, which will store all the experiment results and model checkpoints.

`./prompt` stores the LLM prompts, `./cnndm` stores the experiment data, `./outputs/system` stores the model outputs, and `./outputs/llm` stores the LLM outputs.

## Training
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

#### MLE Training (BART.GPT3D3)
For the MLE training, please run the following command:
```
python main_mle.py --cuda --gpuid [list of gpuid] --config gpt3_mle --model_pt [path to the checkpoint from the warmup training] -l 
```
The checkpoint based on the validation MLE loss (`cache/*/model_mle`) will be used for evaluation.

We also need to train another checkpoint with less training data for the next step:
```
python main_mle.py --cuda --gpuid [list of gpuid] --config gpt3_mle_small --model_pt [path to the checkpoint from the warmup training] -l
```


#### Contrastive Learning (BRIO.GPT3D3)
For the contrastive learning, please run the following command:
```
python main.py --cuda --gpuid [list of gpuid] --config gpt3_brio --model_pt [path to the checkpoint from the MLE training] -l
```
The checkpoint based on the validation contrastive loss (`cache/*/model_ranking`) will be used for evaluation.

### Experiment 2: Learning with GPTRank using ChatGPT

#### MLE Training (BART.ChatGPT)
For the MLE training, please run the following command:
```
python main_mle.py --cuda --gpuid [list of gpuid] --config chatgpt_mle --model_pt [path to the checkpoint from the warmup training] -l 
```
This is similar to the MLE training in the warmup step, but we use more training data here.
The checkpoint based on the validation MLE loss (`cache/*/model_mle`) will be used for evaluation.

#### Contrastive Learning (BRIO.ChatGPT)
For the contrastive learning, please run the following command:
```
python main.py --cuda --gpuid [list of gpuid] --config chatgpt_brio --model_pt [path to the checkpoint from the warmup training] -l
```
The checkpoint based on the validation contrastive loss (`cache/*/model_ranking`) will be used for evaluation.

### Experiment 3: Learning with GPTRank using GPT-4

#### MLE Training (BART.GPT-4)
For the MLE training, please run the following command:
```
python main_mle.py --cuda --gpuid [list of gpuid] --config gpt4_mle --model_pt [path to the checkpoint from the warmup training] -l 
```
The checkpoint based on the validation MLE loss (`cache/*/model_mle`) will be used for evaluation.

#### Contrastive Learning (BRIO.GPT-4)
For the contrastive learning, please run the following command (we will use the checkpoint from the warmup training here):
```
python main.py --cuda --gpuid [list of gpuid] --config gpt4_brio --model_pt [path to the checkpoint from the warmup training] -l
```
The checkpoint based on the validation contrastive loss (`cache/*/model_ranking`) will be used for evaluation.

## Evaluation
### Output Generation
To generate the model outputs, please run the following command:
```
python test.py --gpuid gpuid --src_dir [path to the source article file] --tgt_dir [path to the system output file] 
--ref_dir [path to the reference summary file] --model_dir [path to the model checkpoint] --batch_size BATCH_SIZE --num_beams NUM_BEAMS --length_penalty LENGTH_PENALTY
```
Please note that the length of summaries can have a large impact on the evaluation results ([ref](https://arxiv.org/abs/2212.07981)). Therefore, we tried to maintain the system output length to be similar to the reference summary length, and tune the beam search parameters on the validation set to achieve this goal.

We recommend using the following parameters for the beam search:

| Model | Beam Size | Length Penalty |
| --- | --- | --- |
| BART.GPT3D3 | 4 | 0 |
| BRIO.GPT3D3 | 64 | 0.6 |
| BART.ChatGPT | 4 | 0.8 |
| BRIO.ChatGPT | 64 |  0.8  |
| BART.GPT-4 | 4 | 0.8 | 
| BRIO.GPT-4 | 128 | 0 |

We note that for checkpoints trained with contrastive learning we use a large beam size since previous work has found that a large beam size is beneficial for checkpoints trained with contrastive learning ([ref](https://aclanthology.org/2022.acl-long.207/)).


### LLM-based Evaluations
For LLM-based evaluation, you will need to use OpenAI's APIs. In `gpt.py`, please replace the `openai.organization` and `openai.api_key` with your own credentials.

#### GPTScore
To evaluate the model outputs with GPTScore, please first get the raw scores using the following command:
```
python gpt_score.py -r --src_dir [path to the source article file] --tgt_dir [path to the system output file] --output_dir [path to the output file]
```
The output file will be saved as a jsonl file at the `output_dir`.

Then, please run the following command to get the actual scores:
```
python gpt_score.py -s --length_peanlty LENGTH_PENALTY --output_dir [path to the output file]
```
To get the original GPTScore, please set the length penalty to 1.

#### GPTRank
To evaluate the model outputs with GPTRank (**pair-wise comparison**), please run the following command to get the raw outputs:
```
python gpt_rank.py -r --src_dir [path to the source article file] --cand_dir_1 [path to the system1 output file] --cand_dir_2 [path to the system2 summary file] --tgt_json_dir [path to the output jsonl file] --tgt_txt_dir [path to the output txt file] --model [OpenAI model name]
```
The output file will be saved as a jsonl file at the `tgt_json_dir` and a txt file at the `tgt_txt_dir`.

Then, please run the following command to get the actual scores:
```
python gpt_rank.py -s --tgt_json_dir [path to the output jsonl file] --system_1 [system1 name] --system_2 [system2 name] --output_dir [path to the output file]
```
The output file will be saved as a json file at the `output_dir`.

In addition to the pair-wise comparison, we also provide the code for **list-wise comparison** that is used in our contrastive training. Please refer to the function ``gpt_rank`` in `gpt_rank.py` for more details.

## Data

Please find the training data under the `cnndm` directory. Each subdirectory contains the training data and the validation data for the corresponding model. The input articles used for testing can be found at `cnndm/test.source`.

Here is the description of the different data sets:
- `./chatgpt`: data used for the warmup training
- `./gpt3`: data used for the MLE training with GPT-3, which is used for the next step of contrastive learning
- `./gpt3_all`: data used for the MLE training with GPT-3, which is used for the evaluation
- `./gpt_brio_gptscore`: data used for the contrastive learning with GPT-3 and GPTScore
- `./chatgpt_all`: data used for the MLE training with ChatGPT, which is used for the evaluation
- `./brio_chatgpt`: data used for the contrastive learning with ChatGPT and GPTRank
- `./gpt4`: data used for the MLE training with GPT-4
- `./brio_gpt4`: data used for the contrastive learning with GPT-4 and GPTRank

## Model Outputs

The model outputs on the test set can be found under the `outputs` directory. 
The LLM outputs are in the `outputs/llm` directory, and the system outputs are in the `outputs/system` directory.



