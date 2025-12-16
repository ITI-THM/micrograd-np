"""
Neural network building blocks for micrograd.

This module provides classes to build neural networks with automatic differentiation.
"""

import numpy as np
from micrograd.engine import Value


class Module:
    """
    Base class for all neural network modules.

    Provides common functionality for managing parameters and gradients.
    """

    def zero_grad(self):
        """
        Reset all gradients to zero.

        Call this before each backward pass to avoid accumulating gradients
        from multiple backward passes.
        """
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def parameters(self):
        """
        Return a list of all trainable parameters (weights and biases).

        Override this in subclasses to return actual parameters.
        """
        return []

class Linear(Module):
    """
    Fully-connected (dense) neural network layer.

    Performs the operation: output = activation(x @ W + b)
    Where x is the input, W are the weights, and b is the optional bias.

    Args:
        nin: Number of input features
        nout: Number of output features (neurons)
        nonlin: If True, apply ReLU activation (default: True)
        weights: Optional pre-initialized weights (nin, nout)
        bias: Optional pre-initialized bias (nout,)
        use_bias: Whether to use bias term (default: True)
        name: Optional name for debugging

    Example:
        >>> layer = Linear(3, 5)  # 3 inputs, 5 outputs with ReLU
        >>> x = Value([[1, 2, 3]])  # Batch of 1 sample
        >>> y = layer(x)  # Shape: (1, 5)
    """

    def __init__(self, nin, nout, nonlin=True, weights=None, bias=None, use_bias=True, name=""):
        # Xavier/Glorot uniform initialization for better training
        limit = np.sqrt(6.0 / (nin + nout))
        self.name = name

        # Initialize weights
        if weights is not None:
            self.W = Value(np.array(weights), name=f"W_{self.name}")
        else:
            self.W = Value(np.random.uniform(-limit, limit, (nin, nout)), name=f"W_{self.name}")

        # Initialize bias (optional)
        self.use_bias = use_bias
        if self.use_bias:
            if bias is not None:
                self.b = Value(np.array(bias).reshape(1, nout), name=f"b_{self.name}")
            else:
                self.b = Value(np.zeros((1, nout)), name=f"b_{self.name}")
        else:
            self.b = None

        self.nonlin = nonlin

    def __call__(self, x):
        """
        Forward pass: compute layer output.

        Args:
            x: Input Value with shape (batch_size, nin)

        Returns:
            Output Value with shape (batch_size, nout)
        """
        # Linear transformation: x @ W + b
        act = x @ self.W.T

        if self.use_bias:
            act = act + self.b

        # Apply activation function if specified
        if self.nonlin:
            act = act.relu()

        act.name = self.name
        return act

    def set_weights(self, weights):
        """Set layer weights to specific values."""
        self.W = Value(np.array(weights), name=f"W_{self.name}")

    def set_bias(self, bias):
        """Set layer bias to specific values."""
        if self.use_bias:
            nout = self.W.data.shape[1]
            self.b = Value(np.array(bias).reshape(1, nout), name=f"b_{self.name}")

    def parameters(self):
        """Return list of trainable parameters (weights and bias)."""
        params = [self.W]
        if self.use_bias:
            params.append(self.b)
        return params

    def __repr__(self):
        activation = 'ReLU' if self.nonlin else 'Linear'
        return f"Linear({self.W.data.shape[0]} → {self.W.data.shape[1]}, {activation})"

class MLP(Module):
    """
    Multi-Layer Perceptron: a sequence of fully-connected layers.

    Automatically constructs a neural network with the specified architecture.
    The last layer has no activation (linear output), suitable for regression
    or for adding a final softmax/sigmoid for classification.

    Args:
        nin: Number of input features
        nouts: List of output sizes for each layer
               Example: [16, 16, 1] creates 3 layers: input→16→16→1
        weights: Optional list of pre-initialized weights for each layer
        biases: Optional list of pre-initialized biases for each layer
        use_bias: Whether to use bias terms (default: True)

    Example:
        >>> # Create a 3-layer network: 3 → 16 → 16 → 1
        >>> mlp = MLP(nin=3, nouts=[16, 16, 1])
        >>> x = Value([[1, 2, 3]])  # Input batch
        >>> y = mlp(x)  # Forward pass
        >>> loss = y.mse(target)
        >>> mlp.zero_grad()  # Reset gradients
        >>> loss.backward()  # Compute gradients
        >>> # Update parameters (SGD)
        >>> for p in mlp.parameters():
        ...     p.data -= learning_rate * p.grad
    """

    def __init__(self, nin, nouts, weights=None, biases=None, use_bias=True):
        # Build layer sizes: [input_size, hidden1, hidden2, ..., output_size]
        layer_sizes = [nin] + nouts

        # Create layers: all have ReLU except the last one (linear output)
        self.layers = []
        for i in range(len(nouts)):
            is_output_layer = (i == len(nouts) - 1)

            layer = Linear(
                nin=layer_sizes[i],
                nout=layer_sizes[i + 1],
                nonlin=not is_output_layer,  # No activation on output layer
                weights=weights[i] if weights is not None else None,
                bias=biases[i] if biases is not None else None,
                use_bias=use_bias,
                name=f"layer{i}"
            )
            self.layers.append(layer)

    def __call__(self, x):
        """
        Forward pass: pass input through all layers sequentially.

        Args:
            x: Input Value with shape (batch_size, nin)

        Returns:
            Output Value with shape (batch_size, nouts[-1])
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def set_weights(self, weights):
        """Set weights for all layers."""
        for layer, w in zip(self.layers, weights):
            layer.set_weights(w)

    def set_biases(self, biases):
        """Set biases for all layers."""
        for layer, b in zip(self.layers, biases):
            layer.set_bias(b)

    def parameters(self):
        """Return all trainable parameters from all layers."""
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        layer_str = ' → '.join(str(layer) for layer in self.layers)
        return f"MLP[\n  {layer_str}\n]"
