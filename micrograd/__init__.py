"""
Micrograd: A minimal educational autograd engine.

This package provides automatic differentiation for building and training
neural networks from scratch.
"""

from micrograd.engine import Value
from micrograd import nn
from micrograd.utils import draw_dot

__version__ = "0.1.0"
__all__ = ["Value", "nn", "draw_dot"]
