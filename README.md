# BLOOM for ARC

In this repository is code to train a BLOOM model to answer questions in the [AI2 Reasoning Challenge (ARC) dataset](https://allenai.org/data/arc). Most of the heavy lifting is done by [Hugging Face](https://github.com/huggingface), specifically the [transformers](https://github.com/huggingface/transformers), [datasets](https://github.com/huggingface/datasets), and [peft](https://github.com/huggingface/peft) libraries.

This isn't a comprehensive model/exploration by any means, just a decent starting point that I threw together to enable other experiments I want to do. Similarly, this README is only okay: I provide the basic commands, but you may need to fill in some blanks to do exactly what you want to do.

## General Approach
This repository uses zero or few-shot learning, where the number of shots is set based on a command line argument. Fine-tuning is performed using methods from the [peft](https://github.com/huggingface/peft) library.

Finetuning is simply the prediction loss returned by huggingface. A prediction is considered *valid* if the returned answer is of the format:

`[A, B, C, D, 1, 2, 3, 4]: The matching answer given in the question.`

A prediction is considered *correct* if the answer matches the one in the answer key. In both cases, we end the candidate answer at a line break, strip off \<\/s\>  and \<pad\> tags, remove periods and beginning/ending spaces, and convert to lower. **NB:** I believe the split on the \n is unnecessary, because it looks like the </s> tag is deployed properly, but it continues to generate until some other condition is reached (perhaps until all queries in the batch have terminated.)

## Supported Methods
* As of now, only prefix tuning has been tested, but LoRA and prompt tuning can be enabled via command line. If you try them and they work, let me know. If they don't work, let me know. If you make them work, send a pull request.
* For resource reasons I have only finetuned on the bigscience/bloomz-1b7 model. bigscience/bloom1b7 should work without modification. Larger models have not been tested due to challenges running on multiple GPUs. One would expect performance to improve with the larger models.

## Training
The best model (returned the highest accuracy on val) in our gridsearch can be trained with the command:

`python main.py --base_model bigscience/bloomz-1b7 --batch_size 2 --device cuda --end_lr_divisor 10 --finetune_method prefix --n_shot 0 --split ARC-Easy --start_lr 0.01 --virtual_tokens 45`

This will train for 120 epochs by default, but the best model in our experiment began overfitting after 4 epochs.

## Evaluation
You should be able to run this right away---the weights are part of the git repository:

`python main.py --base_model bigscience/bloomz-1b7 --batch_size 2 --device cuda --finetune_method prefix --n_shot 0 --split ARC-Easy --virtual_tokens 45 --finetuned_model saved_models/bloomz-1b7-prefix-45-0-ARC-Easy-1683219521/best_acc --eval_only`
 
## Performance
The model included with this repository achieves 62.281% accuracy, 99.298% validity, and 62.721% accuracy on valid outputs on the easy split.

On the challenge split, it achieves 40.134% accuracy, 98.328% validity, and 40.816% accuracy on valid examples. The model with the same hyperparameters trained on the challenge set achieves an accuracy of 38.462%. Broadly, this indicates that training on more data is more important than training on so-called hard problems---likely, the distribution shift is not so meaningful from a training standpoint.

The full gridsearch results are [here](gridearch_results.csv). It is worth noting that the prompts for --n_shot > 1 were produced randomly, so there is some stochasticity in performance.

## Installing environment
Environment is packaged with conda:

`conda env create -f environment.yml`
