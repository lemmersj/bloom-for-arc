# BLOOM for ARC

In this repository is code to train a BLOOM model to answer questions in the [AI2 Reasoning Challenge (ARC) dataset] (https://allenai.org/data/arc). Most of the heavy lifting is done by [Hugging Face](https://github.com/huggingface), specifically the [transformers](https://github.com/huggingface/transformers), [datasets](https://github.com/huggingface/datasets), and [peft](https://github.com/huggingface/peft) libraries.

## General Approach
This repository uses zero or few-shot learning, where the number of shots is set based on a command line argument. Fine-tuning is performed using methods from the [peft](https://github.com/huggingface/peft) library.

## Supported Methods
* As of now, only prefix tuning has been tested, but LoRA and prompt tuning can be enabled via command line. If you try them and they work, let me know. If they don't work, let me know. If you make them work, send a pull request.
* For resource reasons I have only finetuned on the bigscience/bloomz-1b7 model. bigscience/bloom1b7 should work without modification. Larger models have not been tested due to challenges running on multiple GPUs. One would expect performance to improve with the larger models.

## Performance

## Gridsearch Results
