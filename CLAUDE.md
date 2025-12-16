# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an **educational implementation of micrograd** - a minimal autograd engine for building and training neural networks. The code is designed for clarity and learning, with extensive documentation explaining the mathematics behind each operation.

Key features:
- Automatic differentiation (backpropagation) for scalars, vectors, and matrices
- NumPy-based implementation with broadcasting support
- Clean, well-documented code suitable for teaching

## Architecture

### Core Components

**engine.py** - Automatic Differentiation Engine (~530 lines)
- `Value` class: Core abstraction that wraps numpy arrays and builds computational graphs
- **Arithmetic operations**: `+`, `-`, `*`, `/`, `**`, `@` (matrix multiplication) - all with gradient support
- **Activation functions**:
  - `relu()`: Rectified Linear Unit
  - `sigmoid()`: Numerically stable sigmoid
  - `softmax()`: Multi-class probability distribution
- **Loss functions**:
  - `mse()`: Mean Squared Error for regression
- **Aggregations**: `sum()`, `mean()` - preserve computational graph
- **Core method**: `backward()` - Computes all gradients via reverse-mode autodiff
- **Helper**: `_handle_broadcast_gradient()` - Handles gradient flow when shapes differ

**nn.py** - Neural Network Building Blocks (~204 lines)
- `Module`: Base class providing `parameters()` and `zero_grad()` methods
- `Linear`: Fully-connected layer with Xavier/Glorot initialization
  - Configurable activation (ReLU or none)
  - Optional bias term
  - Matrix-based for efficiency
- `MLP`: Multi-Layer Perceptron (sequential stack of Linear layers)
  - Last layer is always linear (no activation)
  - Easy to construct: `MLP(nin=3, nouts=[16, 16, 1])`

### Key Design Patterns

1. **Computational Graph**: Each operation creates a new `Value` that tracks its parents (`_prev`) and gradient function (`_backward`)

2. **Operator Overloading**: Python operators automatically build the graph:
   ```python
   z = x * y + b  # Creates graph: x → (*) → (+) → z
                  #                y ↗     ↗
                  #                       b
   ```

3. **Lazy Backpropagation**: Gradients computed only when `.backward()` is called
   - Topological sort ensures parents computed before children
   - Each node's `_backward()` function implements local gradient

4. **Broadcasting**: NumPy broadcasting works in forward pass; gradients automatically summed in backward pass

5. **Educational Comments**: Every backward pass includes docstring explaining the mathematics

## Usage Patterns

### Basic Example: Training a Neural Network

```python
from micrograd.nn import MLP
from micrograd.engine import Value
import numpy as np

# Create a 3-layer network: 3 inputs → 16 hidden → 16 hidden → 1 output
model = MLP(nin=3, nouts=[16, 16, 1])

# Prepare data (must be numpy arrays)
X_train = np.array([[1, 2, 3], [4, 5, 6]])  # 2 samples, 3 features
y_train = np.array([[1.0], [2.0]])           # 2 targets

# Training loop
learning_rate = 0.01
for epoch in range(100):
    # Forward pass
    X = Value(X_train)
    y = Value(y_train)
    predictions = model(X)

    # Compute loss
    loss = predictions.mse(y)

    # Backward pass
    model.zero_grad()      # Reset gradients (important!)
    loss.backward()        # Compute gradients

    # Update parameters (simple SGD)
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")
```

### Understanding the Computational Graph

```python
from micrograd.engine import Value

# Build a simple computation
x = Value(2.0)
y = Value(3.0)
z = x * y + x  # z = 2*3 + 2 = 8

# Compute gradients
z.backward()

print(f"dz/dx = {x.grad}")  # Should be y + 1 = 4.0
print(f"dz/dy = {y.grad}")  # Should be x = 2.0
```

### Important Notes

1. **Data Format**: All data stored as numpy arrays in `Value.data`
2. **Gradient Accumulation**: Gradients accumulate, so call `zero_grad()` before each backward pass
3. **Scalar Loss**: `backward()` must be called on a scalar (use `.mean()` or `.sum()` if needed)
4. **Matrix Shapes**:
   - Input: `(batch_size, features)`
   - Weights: `(features_in, features_out)`
   - Use `.T` for transpose
5. **Activation Functions**: Call as methods (e.g., `x.relu()`, `logits.softmax()`)
6. **No Optimizer Classes**: This is intentional - implement SGD/Adam manually for learning

## Educational Focus

This codebase prioritizes **clarity over performance**:
- Extensive docstrings explain the math (e.g., "d(a*b)/da = b")
- Examples in docstrings show usage
- Comments explain *why*, not just *what*
- Single implementation of each concept (no redundant alternatives)

When modifying:
- Maintain clear documentation
- Add examples to docstrings
- Explain the mathematical reasoning
- Keep code simple and readable
