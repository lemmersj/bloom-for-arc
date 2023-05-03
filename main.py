import argparse
from datasets import load_dataset
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader, Dataset
from peft import PromptTuningConfig, TaskType, PromptTuningInit, get_peft_model, LoraConfig, PrefixTuningConfig
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pdb
import random
import time
import os
"""A simple script for training and evaluating BLOOM on the AI2-ARC dataset."""

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class BloomWrapper():
    """An object containing convenience functions for bloom."""
    def __init__(self, args):
        """Initializes the object, sets self variables, etc.

        args:
            args: the command line argument.

        Returns: nothing.
        """
        self.start_lr = args.start_lr
        self.end_lr_divisor = args.end_lr_divisor
        self.device = args.device

        # Set the finetune method.
        if args.finetune_method == "prompt":
            peft_config = PromptTuningConfig(
                task_type = TaskType.CAUSAL_LM,
                prompt_tuning_init=PromptTuningInit.TEXT,
                num_virtual_tokens=args.virtual_tokens,
                prompt_tuning_init_text=\
                "Please answer the following multiple choice question:\n",
                tokenizer_name_or_path=args.base_model)
        elif args.finetune_method == "lora":
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout)
        elif args.finetune_method == "prefix":
            peft_config = PrefixTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=30)
        else:
            print("Invalid finetune method specified.")
            sys.exit()
        
        # Load the model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(args.base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        self.tokenizer.model_max_length = 256 # This was set in the example

        # Initialize or load the finetuned parameters.
        if args.finetuned_model is not None:
            self.model = Peftmodel.from_pretrained(self.model, args.finetuned_model) 
        else:
            self.model = get_peft_model(self.model, peft_config)

        self.model.to(self.device)
        
        # Set tracking variables.
        self.losses = [] # Training loss
        self.val_correct = [] # Is the answer correct?
        self.answer_valid = [] # Is the answer valid (i.e., in the option set)

    def training_step(self, batch):
        """Peforms a training step.

        args:
            batch: a batch of data.
        
        Returns: loss
        """
        # Tokenize the input.
        tokenized_batch = self.tokenizer(
            batch['full_phrase'], return_tensors="pt", padding="max_length",
            truncation=True)

        # Re-map the information from the tokenizer.
        new_batch = {"input_ids": tokenized_batch['input_ids'].to(self.device),
                     "labels": tokenized_batch['input_ids'].to(self.device),
                     "attention_mask": tokenized_batch['attention_mask'].to(
                         self.device)}

        # Perform the forward pass
        result = self.model(**new_batch)

        self.losses.append(result.loss.detach().cpu())
        # Return the loss
        return result.loss

    def training_epoch_end(self):
        """Ends a training epoch and sends results to wandb.

        args:
            none

        returns: nothing
        """
        wandb.log({"train/loss": np.array(self.losses).mean()})
        self.losses = []

    def validation_step(self, batch):
        """Peforms a single batch of validation.

        args:
            batch: the input data.
        """
        # Tokenize the input batch.
        tokenized_batch = self.tokenizer(
            batch['prompt'], return_tensors="pt", padding="max_length",
            truncation=True)

        # Tokenize the prompt. The maximum number of generated tokens is the
        # size of the longest tokenization minus the size of the prompt.
        tokenized_for_length = self.tokenizer(
            batch['full_phrase'], return_tensors="pt", padding="max_length",
            truncation=True)['attention_mask'].sum(axis=1)
        num_in_prompt = tokenized_batch['attention_mask'].sum(axis=1)
        max_new_tokens = (tokenized_for_length - num_in_prompt).max()

        # Generate the answer.
        result = self.model.generate(
            input_ids = tokenized_batch['input_ids'].to(self.device),
            attention_mask=tokenized_batch['attention_mask'].to(self.device),
            max_new_tokens=max_new_tokens)

        # Evaluate the answers one-by-one
        # TODO: Consider moving the string cleaning to a separate fn.
        for i in range(result.shape[0]):
            # Extract the given answer from the freeform text.
            answer = self.tokenizer.decode(result[i]).split("Answer: ")[-1]
            answer = answer.split("\n")[0]
            answer = answer.replace("</s>", "")
            answer = answer.replace("<pad>", "")
            
            # Is the answer correct?
            self.val_correct.append(
                batch['target'][i].lower().strip().replace(
                    ".", "") == answer.lower().strip().replace(".", ""))

            # Is the answer valid (i.e., does it match a given option?)
            self.answer_valid.append(
                answer.lower().strip().replace(
                    ".", "") in batch['prompt'][i].lower().strip().replace(
                        ".", ""))
    def on_validation_epoch_end(self):
        """Finishes a validation epoch. Logs and resets variables."""
        out_dict = {}
        out_dict["val/acc"] = np.array(self.val_correct).mean()
        out_dict["val/valid"] = np.array(self.answer_valid).mean()

        where_valid = np.where(np.array(self.answer_valid))[0]
        if where_valid.shape[0] > 0:
            out_dict["val/correct_given_valid"] = np.array(
                self.val_correct)[where_valid].mean()

        wandb.log(out_dict)
        self.answer_valid = []
        self.val_correct = []

    def configure_optimizers(self):
        """Gets the optimizers.

        args: none

        returns: optimizer and LR scheduler.
        """
        # TODO: translate parameters to command line args.
        optimizer = AdamW(self.model.parameters(), lr=self.start_lr)
        scheduler = lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=1/self.end_lr_divisor,
            total_iters=120)
        return optimizer, scheduler

class MCQADataset(Dataset):
    """A class meant to simplify multiple choice problems
    
    Note: this is likely not sufficiently general to extend beyond AI2-ARC
    """
    def __init__(self, base_dataset, split, n_shot):
        """Initializes the dataset.

        args:
            base_dataset: a multiple-choice dataset from huggingface hub.
            split: train/val/test
            n_shot: how many examples to give the model as a prefix.
        
        returns: nothing.
        """
        self.n_shot = n_shot

        # Initialize containers for the data.
        self.prompt_strings = [] # The questions that are given to the model.
        self.target_strings = [] # The target outputs.
        self.dataset_strings = [] # The prompt and the target together.

        for item in base_dataset[split]:
            prompt, target, dataset = self.build_question(item)
            self.prompt_strings.append(prompt)
            self.target_strings.append(target)
            self.dataset_strings.append(dataset)

        self.train_strings = [] # Training strings for n-shot learning.
        for item in base_dataset["train"]: # Prompts are only from train dataset.
            _, _, dataset = self.build_question(item)
            self.train_strings.append(dataset)

    def __len__(self):
        """Gets the length of the dataset

        returns; the length of the datset (len(self.dataset_strings))
        """
        return len(self.dataset_strings)

    def __getitem__(self, i):
        """Gets an item from the dataset.

        args:
            i: the index of the dataset item.

        returns: a dict containing the prompt, full phrase, and target.
        """
        prompt_string = ""

        # There may be some contamination during training, validation is safe.
        examples = random.sample(self.train_strings, self.n_shot)
        for example in examples:
            prompt_string += example + "\n\n"

        prompt_string += self.prompt_strings[i]

        to_return = {"prompt": prompt_string,
                     "full_phrase": prompt_string + self.target_strings[i],
                     "target": self.target_strings[i]}

        return to_return

    def build_question(self, item):
        """Builds the properly formatted natural language question.

        args:
            item: the dataset item.

        returns: a triplet containing the question, the answer, and the two
        concatentated.
        """
        base_prompt = ""
        base_prompt += item['question']
        base_prompt += "\n"

        for j in range(len(item['choices']['label'])):
            base_prompt += item['choices']['label'][j]
            base_prompt += ". "
            base_prompt += item['choices']['text'][j]
            base_prompt += "\n"
     
        base_prompt += "\nAnswer: "

        correct_answer = item['answerKey']
        correct_answer_idx = np.where(
            np.array(item['choices']['label'])==correct_answer)[0]
        assert len(correct_answer_idx) == 1
        
        target = f"{item['answerKey']}. {item['choices']['text'][correct_answer_idx.item()]}"

        return (base_prompt, target, base_prompt+target)

if __name__ == "__main__":
    
    # Get and set command line args.
    # TODO: add help.
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_lr", type=float, required=True)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--finetune_method", type=str,
                        choices=["prompt","prefix","lora"], required=True)
    parser.add_argument("--virtual_tokens", type=int)
    parser.add_argument("--lora_r", type=int)
    parser.add_argument("--lora_alpha", type=float)
    parser.add_argument("--lora_dropout", type=float)
    parser.add_argument("--finetuned_model", type=str)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--n_shot", type=int)
    parser.add_argument("--end_lr_divisor", type=float, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--split", type=str, required=True,
                        choices=["ARC-Easy", "ARC-Challenge"])
    args = parser.parse_args()

    model = BloomWrapper(args)
    base_dataset = load_dataset("ai2_arc", args.split)
    val_dataset = MCQADataset(base_dataset, "validation", n_shot=args.n_shot)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=os.cpu_count())
    train_dataset = MCQADataset(base_dataset, "train", n_shot=args.n_shot)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count())
    
    wandb.init(project="tune-bloom", config=args)
    optimizer, scheduler = model.configure_optimizers()
    best_acc = 0
    
    out_dir = f"saved_models/{str(time.time()).split('.')[0]}-{args.base_model.split('/')[-1]}-{args.finetune_method}-{args.virtual_tokens}-{args.n_shot}/"
    for epoch in range(120):
        model.model.eval()
        i = 0
        for eval_data in val_loader:
            model.validation_step(eval_data)
            i+=1
        to_log = {}
        to_log["epoch"] = epoch
        to_log["val/acc"] = np.array(model.val_correct).mean()
        to_log["val/valid"] = np.array(model.answer_valid).mean()
        where_valid = np.where(np.array(model.answer_valid))[0]
        if where_valid.shape[0] > 0:
            to_log["val/correct_given_valid"] = np.array(
                model.val_correct)[where_valid].mean()
        
        # If we're only doing evaluation, print the results and exit.
        if args.eval_only:
            print(to_log)
            sys.exit()
        if to_log["val/acc"] > best_acc:
            best_acc = to_log["val/acc"]
            model.model.save_pretrained(out_dir+"best")
        model.model.save_pretrained(out_dir+"last")

        model.on_validation_epoch_end()
        model.model.train()

        i = 0
        for train_data in train_loader:
            loss = model.training_step(train_data)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            i+=1

        model.training_epoch_end()
        scheduler.step()
