"""
Tokenizer implementation for the Small Language Model
"""

from typing import List, Dict, Optional
import json
from pathlib import Path
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

class Tokenizer:
    def __init__(self, vocab_size: int = 30000, min_frequency: int = 2):
        """
        Initialize the tokenizer
        
        Args:
            vocab_size: Size of the vocabulary
            min_frequency: Minimum frequency for tokens to be included
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.tokenizer = HFTokenizer(BPE())
        self.tokenizer.pre_tokenizer = Whitespace()
        self.trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"]
        )
    
    def train(self, files: List[str]) -> None:
        """
        Train the tokenizer on text files
        
        Args:
            files: List of file paths to train on
        """
        self.tokenizer.train(files, self.trainer)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to token IDs
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(text).ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Convert token IDs to text
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids)
    
    def save(self, save_path: str) -> None:
        """
        Save the tokenizer to disk
        
        Args:
            save_path: Path where to save the tokenizer
        """
        self.tokenizer.save(save_path)
    
    @classmethod
    def load(cls, load_path: str) -> 'Tokenizer':
        """
        Load a tokenizer from disk
        
        Args:
            load_path: Path to load the tokenizer from
            
        Returns:
            Loaded Tokenizer instance
        """
        instance = cls()
        instance.tokenizer = HFTokenizer.from_file(load_path)
        return instance
    
    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size"""
        return self.tokenizer.get_vocab_size()
