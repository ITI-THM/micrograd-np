# micrograd-np

A tiny **NumPy-based** Autograd engine for educational purposes. Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a neural network library with a PyTorch-like API. Unlike the original scalar-only micrograd, this version uses **NumPy arrays** for efficient matrix operations and supports **scalars, vectors, and matrices** with full broadcasting support.

### Key Features

- 🔢 **NumPy-based**: Efficient matrix operations with broadcasting
- 🎓 **Educational**: ~520 lines with extensive documentation explaining the math
- 🔄 **Automatic Differentiation**: Backpropagation through complex computational graphs
- 🧠 **Neural Networks**: Matrix-based Linear layers and MLPs with Xavier initialization
- 📊 **Visualization**: Graphviz integration for computational graph visualization
- ✅ **Tested**: Gradients verified against PyTorch

### Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### Quick Start

#### Basic Operations

```python
from micrograd.engine import Value

# Works with scalars
a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}')  # prints 24.7041
g.backward()
print(f'{a.grad:.4f}')  # prints 138.8338, i.e. dg/da
print(f'{b.grad:.4f}')  # prints 645.5773, i.e. dg/db
```

#### Matrix Operations

```python
from micrograd.engine import Value
import numpy as np

# Matrix multiplication
A = Value([[1, 2], [3, 4]])
B = Value([[5, 6], [7, 8]])
C = A @ B
print(C.data)  # Matrix product

# Broadcasting works automatically
x = Value([[1, 2, 3], [4, 5, 6]])  # 2x3 matrix
b = Value([1, 0, -1])               # 1x3 vector (broadcasts)
y = x + b                           # Element-wise addition with broadcasting
```

### Training a Neural Network

```python
from micrograd.nn import MLP
from micrograd.engine import Value
import numpy as np

# Create a 3-layer network: 3 inputs → 16 hidden → 16 hidden → 1 output
model = MLP(nin=3, nouts=[16, 16, 1])

# Prepare data (batch processing supported!)
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
    model.zero_grad()      # Reset gradients
    loss.backward()        # Compute gradients

    # Update parameters (SGD)
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data:.4f}")
```

### Supported Operations

**Arithmetic**: `+`, `-`, `*`, `/`, `**`, `@` (matrix multiplication)

**Activations**:
- `relu()` - Rectified Linear Unit
- `sigmoid()` - Numerically stable sigmoid
- `softmax(axis)` - Multi-class probability distribution

**Loss Functions**:
- `mse(target)` - Mean Squared Error

**Aggregations**:
- `sum(axis, keepdims)` - Sum along axis
- `mean(axis, keepdims)` - Average along axis

**Other**:
- `.T` - Transpose
- Broadcasting support for all operations

### Visualization

Visualize the computational graph using Graphviz:

```python
from micrograd.engine import Value
from micrograd.utils import draw_dot

x = Value(2.0, name='x')
y = Value(-3.0, name='y')
z = x * y + x
z.name = 'z'
z.backward()

# Create visualization
graph = draw_dot(z)
graph.render('computation_graph')  # Saves as SVG
```

![2d neuron](gout.svg)

### Running Tests

Tests compare gradients against PyTorch to verify correctness:

```bash
# Using uv
PYTHONPATH=. uv run pytest test/

# Or using pytest directly
PYTHONPATH=. pytest test/
```

### Architecture

**`engine.py`** (~520 lines) - Automatic differentiation engine
- `Value` class wraps NumPy arrays and builds computational graphs
- Implements forward and backward pass for all operations
- Handles broadcasting correctly in gradients

**`nn.py`** (~205 lines) - Neural network building blocks
- `Module` - Base class with `parameters()` and `zero_grad()`
- `Linear` - Fully-connected layer with Xavier initialization
- `MLP` - Multi-layer perceptron (stack of Linear layers)

**`utils.py`** (~172 lines) - Visualization utilities
- `draw_dot()` - Creates Graphviz visualizations of computational graphs
- `trace()` - Traverses and collects all nodes in the graph

### Differences from Original Micrograd

1. **NumPy-based**: Uses NumPy arrays instead of scalars
2. **Matrix operations**: Supports `@` for matrix multiplication
3. **Broadcasting**: Proper gradient handling for broadcasted operations
4. **More activations**: Added sigmoid and softmax
5. **Matrix-based layers**: Linear layers instead of scalar Neurons
6. **Batch processing**: Process multiple samples simultaneously
7. **Loss functions**: Built-in MSE loss

### Educational Focus

This codebase prioritizes **clarity over performance**:
- Every backward pass has a docstring explaining the mathematics
- Examples in docstrings show usage patterns
- Comments explain *why*, not just *what*
- Clean, readable code suitable for learning

Perfect for understanding:
- How automatic differentiation works
- How neural networks compute gradients
- The math behind backpropagation
- Building ML frameworks from scratch

### License

MIT
