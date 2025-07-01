"""
Tests for the language model
"""

import pytest
import torch
from slm.model import SmallLanguageModel

def test_model_initialization():
    """
    Test model initialization with default parameters
    """
    model = SmallLanguageModel()
    assert model.embedding.weight.shape[0] == 30000  # Default vocab size
    assert model.embedding.weight.shape[1] == 512    # Default embedding dim

def test_model_forward_pass():
    """
    Test model forward pass
    """
    model = SmallLanguageModel()
    
    # Create dummy input
    batch_size = 2
    seq_length = 8
    input_ids = torch.randint(0, 30000, (batch_size, seq_length))
    
    # Forward pass
    output = model(input_ids)
    
    # Check output shape
    assert output.shape == (batch_size, seq_length, 30000)

def test_model_custom_parameters():
    """
    Test model initialization with custom parameters
    """
    vocab_size = 10000
    embedding_dim = 256
    hidden_dim = 512
    num_layers = 4
    dropout = 0.2
    
    model = SmallLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )
    
    assert model.embedding.weight.shape[0] == vocab_size
    assert model.embedding.weight.shape[1] == embedding_dim
    assert model.transformer.num_encoder_layers == num_layers
    assert model.transformer.num_decoder_layers == num_layers
    assert model.transformer.dropout == dropout

def test_model_device_placement():
    """
    Test model device placement
    """
    if torch.cuda.is_available():
        model = SmallLanguageModel().to('cuda')
        assert next(model.parameters()).device.type == 'cuda'
    else:
        model = SmallLanguageModel().to('cpu')
        assert next(model.parameters()).device.type == 'cpu'

def test_model_save_load(tmp_path):
    """
    Test model saving and loading
    """
    model = SmallLanguageModel()
    model_path = tmp_path / "model.pt"
    
    # Save model
    torch.save(model.state_dict(), model_path)
    
    # Load model
    loaded_model = SmallLanguageModel()
    loaded_model.load_state_dict(torch.load(model_path))
    
    # Check if parameters match
    for param1, param2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.equal(param1, param2)
