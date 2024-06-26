import json
from rich import inspect
import random
from transformers import PreTrainedTokenizerBase

from vllm.transformers_utils.tokenizer import get_tokenizer
from typing import AsyncGenerator, List, Tuple
import numpy as numpy
from string import Template


dataset_size = {"short": 1500, "medium": 3000, "large": 20000}


def construct_rag_prompt(conversations1, rag_type: str):
    conversation = ""
    template_string = """Here is a past conversation between human and an assistant: $conversation \n."""
    if rag_type == "short":
        for discussion in conversations1[0:6]:
            if discussion["from"] == "human":
                conversation += "\n\nhuman:"
                conversation += discussion["value"]
            else:
                conversation += "\ngpt:"
                conversation += discussion["value"]
        conversation += "\nhuman:" + conversations1[6]["value"]
    elif rag_type == "medium":
        for discussion in conversations1[0:20]:
            if discussion["from"] == "human":
                conversation += "\n\nhuman:"
                conversation += discussion["value"]
            else:
                conversation += "\ngpt:"
                conversation += discussion["value"]
        conversation += "\nhuman:" + conversations1[20]["value"]
    elif rag_type == "long":
        for discussion in conversations1[0:100]:
            if discussion["from"] == "human":
                conversation += "\n\nhuman:"
                conversation += discussion["value"]
            else:
                conversation += "\ngpt:"
                conversation += discussion["value"]
        conversation += "\nhuman:" + conversations1[100]["value"]
    result = Template(template_string).substitute(conversation=conversation)
    return result


def prepare_rag_dataset(
    datafile: str, tokenizer_id: str, num_prompts: int, rag_type: str
):
    revised_dataset = []
    combined_dataset = []
    filtered_dataset: List[Tuple[str, int, int]] = []
    with open(datafile, "r") as inputfile:
        dataset = json.load(inputfile)
        # # print(type(dataset))
        if rag_type == "short":
            dataset = [data for data in dataset if len(data["conversations"]) > 6]
        elif rag_type == "medium":
            dataset = [data for data in dataset if len(data["conversations"]) > 20]
        elif rag_type == "large":
            dataset = [data for data in dataset if len(data["conversations"]) > 99]
        dataset = [
            data for data in dataset if data["conversations"][0]["from"] == "human"
        ]
        # sample_indices = random.sample(
        #     range(len(dataset)), 5000)
        # print(len(dataset))
        # dataset = [dataset[i] for i in sample_indices]
        prompts = [
            construct_rag_prompt(data["conversations"], rag_type) for data in dataset
        ]
        tokenizer = get_tokenizer(
            tokenizer_id, trust_remote_code=True
        )
        prompt_token_ids = tokenizer(prompts).input_ids
        if rag_type == "short":
            for i in range(len(dataset)):
                revised_dataset.append((prompts[i], prompt_token_ids[i], 500))
        elif rag_type == "medium":
            for i in range(len(dataset)-1):
                revised_dataset.append((prompts[i]+prompts[i+1],prompt_token_ids[i]+prompt_token_ids[i+1],500))
            revised_dataset.append((prompts[-1],prompt_token_ids[-1],500))
        print(f"size of revised dataset is: {len(revised_dataset)}")
        for prompt, prompt_token_ids, completion_len in revised_dataset:
            prompt_len = len(prompt_token_ids)
            if prompt_len < dataset_size[rag_type]:
                continue
            filtered_dataset.append((prompt, prompt_len, completion_len))
        sample_indice = random.sample(range(len(filtered_dataset)),num_prompts)
        filtered_dataset = [filtered_dataset[i] for i in sample_indice]
        print(f"size of final filtered dataset: {len(filtered_dataset)}")
        return filtered_dataset


if __name__ == "__main__":
    dataset = prepare_rag_dataset(
        "./ShareGPT_V3_unfiltered_cleaned_split.json",
        "mistralai/Mistral-7B-Instruct-v0.2",
        1000,
        "short",
    )
    print(len(dataset))
    print(type(dataset[0]))
    print("\nsample:\n")
    print(dataset[400][2])
