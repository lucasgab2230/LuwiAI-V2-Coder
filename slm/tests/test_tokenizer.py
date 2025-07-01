"""
Tests for the tokenizer
"""

import os
import tempfile
from pathlib import Path

import pytest

from slm.tokenizer import Tokenizer


def test_tokenizer_initialization():
    """
    Test tokenizer initialization
    """
    tokenizer = Tokenizer()
    assert tokenizer.vocab_size == 30000
    assert tokenizer.min_frequency == 2


def test_tokenizer_training(tmp_path):
    """
    Test tokenizer training
    """
    # Create temporary text file
    text = "This is a test sentence. Another test sentence."
    temp_file = tmp_path / "test.txt"
    temp_file.write_text(text)

    # Train tokenizer
    tokenizer = Tokenizer()
    tokenizer.train([str(temp_file)])

    # Test encoding
    encoded = tokenizer.encode("test sentence")
    assert isinstance(encoded, list)
    assert all(isinstance(x, int) for x in encoded)

    # Test decoding
    decoded = tokenizer.decode(encoded)
    assert isinstance(decoded, str)


def test_tokenizer_save_load(tmp_path):
    """
    Test tokenizer saving and loading
    """
    # Train tokenizer
    tokenizer = Tokenizer()

    # Save tokenizer
    save_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(save_path))

    # Load tokenizer
    loaded_tokenizer = Tokenizer.load(str(save_path))

    # Test encoding/decoding consistency
    text = "test sentence"
    original_encoded = tokenizer.encode(text)
    loaded_encoded = loaded_tokenizer.encode(text)

    assert original_encoded == loaded_encoded


def test_tokenizer_special_tokens():
    """
    Test special token handling
    """
    tokenizer = Tokenizer()

    # Test special tokens
    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
    for token in special_tokens:
        assert tokenizer.tokenizer.token_to_id(token) is not None

    # Test unknown token
    unknown_token = "xyz123"
    unk_id = tokenizer.tokenizer.token_to_id("<unk>")
    assert tokenizer.tokenizer.token_to_id(unknown_token) == unk_id


def test_tokenizer_batch_processing():
    """
    Test tokenizer with batch processing
    """
    tokenizer = Tokenizer()
    texts = [
        "This is a test sentence.",
        "Another test sentence.",
        "Batch processing test.",
    ]

    # Test batch encoding
    encoded = [tokenizer.encode(text) for text in texts]
    assert len(encoded) == len(texts)

    # Test batch decoding
    decoded = [tokenizer.decode(tokens) for tokens in encoded]
    assert len(decoded) == len(texts)
