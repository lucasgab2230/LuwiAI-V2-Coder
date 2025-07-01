"""
Main entry point for the Small Language Model
"""

import torch
from torch.utils.data import Dataset

from slm.model import SmallLanguageModel
from slm.tokenizer import Tokenizer
from slm.trainer import Trainer


class TextDataset(Dataset):
    """
    Dataset class for text data
    """

    def __init__(self, tokenizer: Tokenizer, texts: List[str], max_length: int = 128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text)

        # Add padding if needed
        if len(tokens) < self.max_length:
            tokens = tokens + [self.tokenizer.tokenizer.token_to_id("<pad>")] * (
                self.max_length - len(tokens)
            )
        else:
            tokens = tokens[: self.max_length]

        # Create input and target sequences
        input_ids = tokens[:-1]
        target_ids = tokens[1:]

        return {
            "input_ids": torch.tensor(input_ids),
            "target_ids": torch.tensor(target_ids),
        }


def train_model():
    """
    Example training function
    """
    # Initialize components
    model = SmallLanguageModel()
    tokenizer = Tokenizer()

    # Example training data
    texts = [
        "This is an example sentence.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
    ]

    # Create dataset and dataloader
    train_dataset = TextDataset(tokenizer, texts)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=True
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Train for 3 epochs
    for epoch in range(3):
        train_metrics = trainer.train_epoch()
        print(f"Epoch {epoch + 1}: {train_metrics}")


def generate_text():
    """
    Example text generation function
    """
    # Initialize model and tokenizer
    model = SmallLanguageModel()
    tokenizer = Tokenizer()

    # Example prompt
    prompt = "Once upon a time"

    # Tokenize input
    input_ids = torch.tensor([tokenizer.encode(prompt)])

    # Generate text
    # Note: This is a placeholder - actual generation logic needs to be implemented
    generated_ids = model.generate(input_ids)
    generated_text = tokenizer.decode(generated_ids)

    print(f"Generated text: {generated_text}")


if __name__ == "__main__":
    # Example usage
    print("\n=== Training Example ===")
    train_model()

    print("\n=== Generation Example ===")
    generate_text()
