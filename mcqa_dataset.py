
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import pdb
import numpy as np

class MCQADataset(Dataset):
    """A class meant to simplify multiple choice problems

    Note: this is likely not sufficiently general to extend beyond AI2-ARC
    """

    def __init__(self, split, n_shot):
        """Initializes the dataset.

        args:
            base_dataset: a multiple-choice dataset from huggingface hub.
            split: train/val/test
            n_shot: how many examples to give the model as a prefix.

        returns: nothing.
        """
        all_datasets = {}
        all_datasets["arc_easy"] = load_dataset("ai2_arc", "ARC-Easy")[split]
        all_datasets["arc_challenge"] = load_dataset("ai2_arc", "ARC-Challenge")[split]
        all_datasets["race"] = load_dataset("race", "all")[split]
        all_datasets["quail"] = load_dataset("quail")[split]
        all_datasets["cosmos_qa"] = load_dataset("cosmos_qa")[split]
        self.n_shot = n_shot

        # Initialize containers for the data.
        self.prompt_strings = []  # The questions that are given to the model.
        self.target_strings = []  # The target outputs.
        self.lm_target_strings = []  # The prompt and the target together.
        
        for dataset in all_datasets:
            print(dataset)
            for item in all_datasets[dataset]:
                prompt, target, lm_target = self.build_question(item, dataset)
                self.prompt_strings.append(prompt)
                self.target_strings.append(target)
                self.lm_target_strings.append(lm_target)

    def __len__(self):
        """Gets the length of the dataset

        returns; the length of the datset (len(self.dataset_strings))
        """
        return len(self.lm_target_strings)

    def __getitem__(self, i):
        """Gets an item from the dataset.

        args:
            i: the index of the dataset item.

        returns: a dict containing the prompt, full phrase, and target.
        """
        prompt_string = ""

        # There may be some contamination during training, validation is safe.
        #examples = random.sample(self.train_strings, self.n_shot)
        #for example in examples:
        #    prompt_string += example + "\n\n"

        prompt_string += self.prompt_strings[i]

        to_return = {
            "prompt": prompt_string,
            "full_phrase": prompt_string + self.target_strings[i],
            "target": self.lm_target_strings[i],
        }

        return to_return

    def build_question(self, item, dataset):
        """Builds the properly formatted natural language question.

        args:
            item: the dataset item.

        returns: a triplet containing the question, the answer, and the two
        concatentated.
        """
        base_prompt = ""
        alphabet = ["A", "B", "C", "D", "E", "F", "G"]
        if "arc" in dataset:
            base_prompt += item["question"]
            base_prompt += "\n"
            for j in range(len(item["choices"]["label"])):
                base_prompt += item["choices"]["label"][j]
                base_prompt += ". "
                base_prompt += item["choices"]["text"][j]
                base_prompt += "\n"
            correct_answer = item["answerKey"]
            correct_answer_idx = np.where(
                np.array(item["choices"]["label"]) == correct_answer
            )[0]
            assert len(correct_answer_idx) == 1
            target = (
                f"{item['answerKey']}. {item['choices']['text'][correct_answer_idx.item()]}"
            )
        elif "race" in dataset:
            base_prompt += item['article']
            base_prompt += "\n\n"
            base_prompt += item["question"]
            base_prompt += "\n"
            for j in range(len(item["options"])):
                base_prompt += alphabet[j]
                base_prompt += ". "
                base_prompt += item["options"][j]
                base_prompt += "\n"

            correct_answer = item['answer']

            correct_answer_idx = np.where(np.array(alphabet) == correct_answer) 
            assert len(correct_answer_idx) == 1
            target = (
                f"{item['answer']}. {item['options'][correct_answer_idx[0][0]]}"
            )
        elif "quail" in dataset:
            base_prompt += item['context']
            base_prompt += "\n\n"
            base_prompt += item["question"]
            base_prompt += "\n"
            for j in range(len(item["answers"])):
                base_prompt += alphabet[j]
                base_prompt += ". "
                base_prompt += item["answers"][j]
                base_prompt += "\n"
            target = (
                f"{alphabet[item['correct_answer_id']]}. {item['answers'][item['correct_answer_id']]}"
            )
        elif "cosmos" in dataset:
            base_prompt += item['context']
            base_prompt += "\n\n"
            base_prompt += item["question"]
            base_prompt += "\n"
            answer_keys = ['answer0','answer1','answer2','answer3']
            for j in range(len(answer_keys)):
                base_prompt += alphabet[j]
                base_prompt += ". "
                base_prompt += item[answer_keys[j]]
                base_prompt += "\n"
            target = (
                f"{alphabet[item['label']]}. {item[answer_keys[item['label']]]}"
            )
        base_prompt += "\nAnswer: "
        print(base_prompt + target)
        return (base_prompt, target, base_prompt + target)

