"""A simple script for training and evaluating BLOOM on the AI2-ARC dataset."""
import argparse
import random
import time
import os
import sys

from torch.optim import AdamW, lr_scheduler
from peft import (
    PromptTuningConfig,
    TaskType,
    PromptTuningInit,
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    PeftModel,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from mcqa_dataset import MCQADataset
import numpy as np
import wandb
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BloomWrapper:
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
                task_type=TaskType.CAUSAL_LM,
                prompt_tuning_init=PromptTuningInit.TEXT,
                num_virtual_tokens=args.virtual_tokens,
                prompt_tuning_init_text="Please answer the following multiple choice question:\n",
                tokenizer_name_or_path=args.base_model,
            )
        elif args.finetune_method == "lora":
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
            )
        elif args.finetune_method == "prefix":
            peft_config = PrefixTuningConfig(
                task_type=TaskType.CAUSAL_LM, num_virtual_tokens=30
            )
        else:
            print("Invalid finetune method specified.")
            sys.exit()

        # Load the model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(args.base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        self.tokenizer.model_max_length = 256  # This was set in the example

        # Initialize or load the finetuned parameters.
        if args.finetuned_model is not None:
            self.model = PeftModel.from_pretrained(self.model, args.finetuned_model)
        else:
            self.model = get_peft_model(self.model, peft_config)

        self.model.to(self.device)

        # Set tracking variables.
        self.losses = []  # Training loss
        self.val_correct = []  # Is the answer correct?
        self.answer_valid = []  # Is the answer valid (i.e., in the option set)

    def training_step(self, batch):
        """Peforms a training step.

        args:
            batch: a batch of data.

        Returns: loss
        """
        # Tokenize the input.
        tokenized_batch = self.tokenizer(
            batch["full_phrase"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        # Re-map the information from the tokenizer.
        new_batch = {
            "input_ids": tokenized_batch["input_ids"].to(self.device),
            "labels": tokenized_batch["input_ids"].to(self.device),
            "attention_mask": tokenized_batch["attention_mask"].to(self.device),
        }

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

    def validation_step(self, batch, print_eval):
        """Peforms a single batch of validation.

        args:
            batch: the input data.
            print_eval: print the input/output strings
        """
        # Tokenize the input batch.
        tokenized_batch = self.tokenizer(
            batch["prompt"], return_tensors="pt", padding="max_length", truncation=True
        )

        # Tokenize the prompt. The maximum number of generated tokens is the
        # size of the longest tokenization minus the size of the prompt.
        tokenized_for_length = self.tokenizer(
            batch["full_phrase"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )["attention_mask"].sum(axis=1)
        num_in_prompt = tokenized_batch["attention_mask"].sum(axis=1)
        max_new_tokens = (tokenized_for_length - num_in_prompt).max()

        # Generate the answer. Added some to max_new_tokens due to observed
        # errors. TODO: Figure out why this is happening.
        result = self.model.generate(
            input_ids=tokenized_batch["input_ids"].to(self.device),
            attention_mask=tokenized_batch["attention_mask"].to(self.device),
            max_new_tokens=max_new_tokens + 5,
        )

        # Evaluate the answers one-by-one
        # TODO: Consider moving the string cleaning to a separate fn.
        for i in range(result.shape[0]):
            # Extract the given answer from the freeform text.
            answer = self.tokenizer.decode(result[i]).split("Answer: ")[-1]
            answer = answer.split("\n")[0]
            answer = answer.replace("</s>", "")
            answer = answer.replace("<pad>", "")
            answer = answer.strip()

            if print_eval:
                print(batch["prompt"][i])
                print(f"Target: {batch['target'][i]}, Guess: {answer}")
                print("---")

            # Is the answer correct?
            self.val_correct.append(
                batch["target"][i].lower().strip().replace(".", "")
                == answer.lower().strip().replace(".", "")
            )

            # Is the answer valid (i.e., does it match a given option?)
            self.answer_valid.append(
                answer.lower().strip().replace(".", "")
                in batch["prompt"][i].lower().strip().replace(".", "")
            )

    def on_validation_epoch_end(self):
        """Finishes a validation epoch. Logs and resets variables.

        return: the logging dict.
        """
        out_dict = {}
        out_dict["val/acc"] = np.array(self.val_correct).mean()
        out_dict["val/valid"] = np.array(self.answer_valid).mean()

        where_valid = np.where(np.array(self.answer_valid))[0]
        if where_valid.shape[0] > 0:
            out_dict["val/correct_given_valid"] = np.array(self.val_correct)[
                where_valid
            ].mean()

        try:
            wandb.log(out_dict)
        except wandb.Error:
            pass

        self.answer_valid = []
        self.val_correct = []

        return out_dict

    def configure_optimizers(self):
        """Gets the optimizers.

        args: none

        returns: optimizer and LR scheduler.
        """
        # TODO: translate parameters to command line args.
        optimizer = AdamW(self.model.parameters(), lr=self.start_lr)
        scheduler = lr_scheduler.LinearLR(
            optimizer,
            start_factor=1,
            end_factor=1 / self.end_lr_divisor,
            total_iters=120,
        )
        return optimizer, scheduler

if __name__ == "__main__":
    # Get and set command line args.
    # TODO: add help.
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_lr", type=float)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument(
        "--finetune_method",
        type=str,
        choices=["prompt", "prefix", "lora"],
        required=True,
    )
    parser.add_argument("--virtual_tokens", type=int)
    parser.add_argument("--lora_r", type=int)
    parser.add_argument("--lora_alpha", type=float)
    parser.add_argument("--lora_dropout", type=float)
    parser.add_argument("--finetuned_model", type=str)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--n_shot", type=int)
    parser.add_argument("--end_lr_divisor", type=float)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--print_eval", action="store_true", default=False)
    parser.add_argument(
        "--split", type=str, required=True, choices=["ARC-Easy", "ARC-Challenge"]
    )
    args = parser.parse_args()

    if args.n_shot > 0:
        print("N-shot learning currently disabled.")
        sys.exit()

    model = BloomWrapper(args)
    val_dataset = MCQADataset("validation", n_shot=args.n_shot)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
    )
    train_dataset = MCQADataset("train", n_shot=args.n_shot)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
    )

    if not args.eval_only:
        wandb.init(project="tune-bloom", config=args)
        optimizer, scheduler = model.configure_optimizers()
        best_acc = 0
        best_valid = 0
        best_acc_given_valid = 0

        out_dir = (
            f"saved_models/{args.base_model.split('/')[-1]}-"
            f"{args.finetune_method}-{args.virtual_tokens}-{args.n_shot}-"
            f"{args.split}-{str(time.time()).split('.', maxsplit=1)[0]}/"
        )
    for epoch in range(120):
        model.model.eval()
        i = 0
        for eval_data in tqdm(val_loader):
            model.validation_step(eval_data, args.print_eval)
            i += 1
            if epoch == 0 and i > 5:
                break
        to_log = model.on_validation_epoch_end()

        # If we're only doing evaluation, print the results and exit.
        if args.eval_only:
            print(to_log)
            sys.exit()
        if to_log["val/acc"] > best_acc:
            best_acc = to_log["val/acc"]
            model.model.save_pretrained(out_dir + "best_acc")
            wandb.log({"val/best_acc": best_acc})
        if to_log["val/valid"] > best_valid:
            best_valid = to_log["val/valid"]
            model.model.save_pretrained(out_dir + "best_valid")
            wandb.log({"val/best_valid": best_valid})
        if to_log["val/correct_given_valid"] > best_acc_given_valid:
            best_acc_given_valid = to_log["val/correct_given_valid"]
            model.model.save_pretrained(out_dir + "best_given_valid")
            wandb.log({"val/best_acc_given_valid": best_acc_given_valid})

        model.model.save_pretrained(out_dir + "last")

        model.model.train()

        i = 0
        for train_data in tqdm(train_loader):
            loss = model.training_step(train_data)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            i += 1
            if i % 200 == 0: 
                model.training_epoch_end()
        wandb.log({"train/epoch": epoch, "train/lr": scheduler.get_last_lr()[0]})
        scheduler.step()
