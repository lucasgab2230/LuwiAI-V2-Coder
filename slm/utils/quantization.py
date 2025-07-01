"""
Quantization utilities for the Small Language Model
"""

import platform
from typing import Any, Dict, Optional

import cpuinfo
import intel_extension_for_pytorch as ipex
import torch
import torch.nn as nn


class Quantizer:
    def __init__(self, model: nn.Module):
        """
        Initialize the quantizer

        Args:
            model: The model to be quantized
        """
        self.model = model
        self.cpu_info = cpuinfo.get_cpu_info()

    def _check_cpu_compatibility(self) -> bool:
        """
        Check if the CPU is compatible with Intel optimizations

        Returns:
            bool: True if CPU is compatible, False otherwise
        """
        # Check if CPU is Intel
        if "Intel" not in self.cpu_info.get("vendor_id", ""):
            return False

        # Check if CPU supports AVX2 (required for some optimizations)
        if "avx2" not in self.cpu_info.get("flags", []):
            return False

        return True

    def quantize_model(self, dtype: torch.dtype = torch.qint8) -> nn.Module:
        """
        Quantize the model for Intel CPU

        Args:
            dtype: Quantization data type

        Returns:
            Quantized model
        """
        if not self._check_cpu_compatibility():
            raise RuntimeError("CPU is not compatible with Intel optimizations")

        # Move model to CPU if not already there
        self.model = self.model.to("cpu")

        # Apply Intel optimizations
        self.model = ipex.optimize(self.model)

        # Quantize the model
        with torch.no_grad():
            quantized_model = torch.quantization.quantize_dynamic(
                self.model, {nn.Linear}, dtype=dtype  # Quantize only Linear layers
            )

        return quantized_model

    def benchmark_quantization(
        self, input_shape: tuple, num_runs: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark the quantization performance

        Args:
            input_shape: Shape of the input tensor
            num_runs: Number of benchmark runs

        Returns:
            Dictionary containing benchmark results
        """
        import time

        # Create dummy input
        dummy_input = torch.randn(input_shape)

        # Benchmark original model
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = self.model(dummy_input)
        original_time = time.time() - start_time

        # Quantize and benchmark quantized model
        quantized_model = self.quantize_model()
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = quantized_model(dummy_input)
        quantized_time = time.time() - start_time

        return {
            "original_time": original_time,
            "quantized_time": quantized_time,
            "speedup": original_time / quantized_time,
            "cpu_info": self.cpu_info,
        }

    def save_quantized_model(self, path: str) -> None:
        """
        Save the quantized model

        Args:
            path: Path to save the quantized model
        """
        quantized_model = self.quantize_model()
        torch.save(quantized_model.state_dict(), path)

    @classmethod
    def load_quantized_model(cls, path: str, model: nn.Module) -> nn.Module:
        """
        Load a quantized model

        Args:
            path: Path to load the quantized model from
            model: Original model architecture

        Returns:
            Loaded quantized model
        """
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        quantized_model.load_state_dict(torch.load(path))
        return quantized_model
