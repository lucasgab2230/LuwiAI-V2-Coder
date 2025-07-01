"""
Trainer implementation for the Small Language Model with quantization support
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from tqdm import tqdm
import os
from pathlib import Path
from slm.utils.quantization import Quantizer

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        quantize: bool = False
    ):  # Added quantize parameter
        """
        Initialize the trainer
        
        Args:
            model: The language model to train
            train_dataloader: DataLoader for training data
            val_dataloader: Optional DataLoader for validation data
            device: Device to train on ("cuda" or "cpu")
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        
        # Initialize optimizer and loss function
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize quantizer if requested
        self.quantizer = None
        if quantize and device == "cpu":
            self.quantizer = Quantizer(model)
        self.loss_fn = nn.CrossEntropyLoss()
        
    def train_epoch(self) -> Dict[str, float]:
        """
        Train the model for one epoch
        
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        total_loss = 0
        total_tokens = 0
        
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            # Forward pass
            output = self.model(input_ids)
            
            # Flatten the output and targets for loss calculation
            output = output.view(-1, output.size(-1))
            target_ids = target_ids.view(-1)
            
            # Calculate loss
            loss = self.loss_fn(output, target_ids)
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / total_tokens
        return {"train_loss": avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model
        
        Returns:
            Dictionary containing validation metrics
        """
        if self.val_dataloader is None:
            return {}
            
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                
                output = self.model(input_ids)
                
                output = output.view(-1, output.size(-1))
                target_ids = target_ids.view(-1)
                
                loss = self.loss_fn(output, target_ids)
                
                total_loss += loss.item() * input_ids.numel()
                total_tokens += input_ids.numel()
        
        avg_loss = total_loss / total_tokens
        return {"val_loss": avg_loss}
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Save a model checkpoint with quantization support
        
        Args:
            path: Path to save the checkpoint
            epoch: Current epoch number
            metrics: Dictionary of metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Save quantized model if quantizer is available
        if self.quantizer:
            quantized_path = path.replace('.pt', '_quantized.pt')
            self.quantizer.save_quantized_model(quantized_path)
            checkpoint['quantized_model_path'] = quantized_path
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        """
        Save a model checkpoint
        
        Args:
            path: Path to save the checkpoint
            epoch: Current epoch number
            metrics: Dictionary of metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load a model checkpoint with quantization support
        
        Args:
            path: Path to load the checkpoint from
            
        Returns:
            Dictionary containing checkpoint information
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load quantized model if available
        if 'quantized_model_path' in checkpoint and self.quantizer:
            quantized_path = checkpoint['quantized_model_path']
            if os.path.exists(quantized_path):
                self.model = self.quantizer.load_quantized_model(
                    quantized_path,
                    self.model
                )
                print(f"Loaded quantized model from {quantized_path}")
        
        return checkpoint
        """
        Load a model checkpoint
        
        Args:
            path: Path to load the checkpoint from
            
        Returns:
            Dictionary containing checkpoint information
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint
