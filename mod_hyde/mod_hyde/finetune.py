from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from datasets import Dataset
import pandas as pd
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
import glob
import json
from dotenv import load_dotenv, find_dotenv
import os
import wandb

_ = load_dotenv(find_dotenv(),override=True)

wandb.login(key=os.environ['WANDB_API_KEY'])

# parser = argparse.ArgumentParser(description='Finetune the model')
# parser.add_argument("-m","--model", type=str,help="Name of the model for finetuning on the raw texts",default="distilbert/distilgpt2")
# parser.add_argument("-bls","--block_size", type=int, help="Block size for the finetuning",default=1024)
# parser.add_argument("-cs","--chunk_size",type=int, help="Chunksize of the raw texts",default=5000)
# parser.add_argument("-co","--chunk_overlap",type=int, help="Chunksize overlap of the raw texts",default=100)
# parser.add_argument("-tbs","--tokenizer_batch_size",type=int, help="Batch size of tokenization",default=2000)
# parser.add_argument("-np","--num_proc",type=int, help="number of processes",default=4)
# parser.add_argument("-bs","--batch_size",type=int, help="Batch size of finetuning",default=4)
# parser.add_argument("-nte","--num_train_epochs",type=int, help="Number of training epochs",default=1)
# parser.add_argument("-lr","--learning_rate",type=float, help="Learning rate",default=2e-5)

# args = parser.parse_args()

os.environ["WANDB_PROJECT"] = "Modified HyDE"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

def split_text_langchain(text,chunk_size:int=5000,chunk_overlap:int=100):
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    splitted_text = text_splitter.split_text(text)
    return splitted_text

def get_hf_dataset(all_text_list:List[str]):
    df = pd.DataFrame(all_text_list,columns=["text"])
    hf_dataset = Dataset.from_pandas(df)
    return hf_dataset


def tokenize_dataset(all_text_list:List[str],tokenizer):
    hf_dataset = get_hf_dataset(all_text_list)
    def tokenize_function(examples):
        return tokenizer(examples["text"])
    def group_texts(examples):
        # Concatenate all texts.
        examples = tokenize_function(examples)
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    # tokenized_datasets = hf_dataset.map(tokenize_function, batched=True,  remove_columns=["text"])
    lm_datasets = hf_dataset.map(
        group_texts,
        batched=True,
        batch_size=2000,
        remove_columns=["text"]
    )
    return lm_datasets

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def finetune_distilgpt2(model_name:str="distilbert/distilgpt2",BLOCK_SIZE:int=512,num_train_epochs:int=3,per_device_train_batch_size:int=8,learning_rate:float=2e-5):
    global block_size
    block_size = BLOCK_SIZE

    # free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
    # max_memory = f"{free_in_GB-2}GB"

    # n_gpus = torch.cuda.device_count()
    # max_memory = {i: max_memory for i in range(n_gpus)}
    # max_memory["cpu"] = "100GB"
    with open("data/all_book_data.txt") as f:
        book_data = f.read()
    splitted_books_data = split_text_langchain(book_data)    
    json_files = glob.glob(f"data/chunked_*.json")

    transcripts_text = []

    for jf in json_files:
        with open(f'{jf}', 'r') as f:
            data = json.load(f)
        for _,texts in data.items():
            for text in texts:
                curr_text = text['text']
                # Less than 100 words, don't consider
                if len(curr_text.split()) <=100: continue
                transcripts_text.append(curr_text)
    all_text_list = splitted_books_data + transcripts_text
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    tokenized_datasets = tokenize_dataset(all_text_list,tokenizer)
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )
    training_args = TrainingArguments(
        output_dir=f"distilbert-base",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        fp16 = not torch.cuda.is_bf16_supported(),
        learning_rate=learning_rate,
        bf16 = torch.cuda.is_bf16_supported(),
        weight_decay=0.01,
        lr_scheduler_type= 'cosine',
        optim = "adamw",
        seed = 42,
        gradient_accumulation_steps = 1,
        report_to="wandb",
        gradient_checkpointing=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets,
        data_collator=data_collator,
    )
    
    trainer.train()

    trainer.save_model(f"distilbert-base")

    return trainer