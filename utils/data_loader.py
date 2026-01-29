"""Data loading utilities for training."""

import os
import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import PreTrainedTokenizer


class TextDataset(Dataset):
    """Simple text dataset for language modeling."""
    
    def __init__(self, texts: List[str], tokenizer: PreTrainedTokenizer, max_length: int = 2048):
        """
        Initialize dataset.
        
        Args:
            texts: List of text strings
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize with truncation and padding
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # For causal LM, labels are the same as input_ids
        labels = input_ids.clone()
        
        # Mask padding tokens in labels (-100 is ignored by loss)
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_texts_from_folder(data_folder: str) -> List[str]:
    """
    Load all text data from a folder.
    
    Args:
        data_folder: Path to data folder
    
    Returns:
        List of text strings
    """
    data_path = Path(data_folder)
    texts = []
    
    if not data_path.exists():
        print(f"Warning: Data folder {data_folder} does not exist!")
        return texts
    
    # Load .txt files
    for txt_file in data_path.glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                texts.append(content)
    
    # Load .json files (expecting {"text": "..."} format)
    for json_file in data_path.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "text" in item:
                        texts.append(item["text"])
                    elif isinstance(item, str):
                        texts.append(item)
            elif isinstance(data, dict) and "text" in data:
                texts.append(data["text"])
    
    # Load .jsonl files
    for jsonl_file in data_path.glob("*.jsonl"):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                if isinstance(item, dict) and "text" in item:
                    texts.append(item["text"])
                elif isinstance(item, str):
                    texts.append(item)
    
    print(f"Loaded {len(texts)} text samples from {data_folder}")
    return texts


def load_training_data(
    data_folder: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    max_length: int = 2048,
    num_workers: int = 0,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """
    Load training data and create DataLoader.
    
    Args:
        data_folder: Path to data folder
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size per GPU
        max_length: Maximum sequence length
        num_workers: Number of data loading workers
        rank: Current process rank (for distributed training)
        world_size: Total number of processes
    
    Returns:
        DataLoader
    """
    # Load texts
    texts = load_texts_from_folder(data_folder)
    
    if not texts:
        raise ValueError(f"No data found in {data_folder}")
    
    # Create dataset
    dataset = TextDataset(texts, tokenizer, max_length)
    
    # Create sampler for distributed training
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader


if __name__ == "__main__":
    # Test data loading
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataloader = load_training_data(
        data_folder="data",
        tokenizer=tokenizer,
        batch_size=2,
        max_length=512,
    )
    
    print(f"\nDataLoader test successful!")
    print(f"Number of batches: {len(dataloader)}")
    
    # Test one batch
    batch = next(iter(dataloader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Input shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
