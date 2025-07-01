"""
Utility functions for data preprocessing
"""

import torch
from typing import List, Dict, Any
import re
from pathlib import Path
import logging

def clean_text(text: str) -> str:
    """
    Clean text by removing unwanted characters and normalizing whitespace
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    # Normalize quotes
    text = re.sub(r"[\"\']", "'", text)
    
    return text.strip()

def batch_texts(texts: List[str], batch_size: int) -> List[List[str]]:
    """
    Batch texts into smaller chunks
    
    Args:
        texts: List of texts
        batch_size: Size of each batch
        
    Returns:
        List of text batches
    """
    return [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

def save_text_dataset(
    texts: List[str],
    output_dir: str,
    filename: str = "dataset.txt"
) -> None:
    """
    Save text dataset to file
    
    Args:
        texts: List of texts to save
        output_dir: Directory to save the dataset
        filename: Name of the output file
    """
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')

def load_text_dataset(
    input_path: str,
    clean: bool = True
) -> List[str]:
    """
    Load text dataset from file
    
    Args:
        input_path: Path to the dataset file
        clean: Whether to clean the text
        
    Returns:
        List of texts
    """
    texts = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if clean:
                text = clean_text(text)
            if text:
                texts.append(text)
    
    return texts

def split_dataset(
    texts: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Dict[str, List[str]]:
    """
    Split dataset into train, validation, and test sets
    
    Args:
        texts: List of texts
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        
    Returns:
        Dictionary containing train, validation, and test sets
    """
    import random
    random.shuffle(texts)
    
    train_size = int(len(texts) * train_ratio)
    val_size = int(len(texts) * val_ratio)
    
    train_texts = texts[:train_size]
    val_texts = texts[train_size:train_size + val_size]
    test_texts = texts[train_size + val_size:]
    
    return {
        'train': train_texts,
        'val': val_texts,
        'test': test_texts
    }
