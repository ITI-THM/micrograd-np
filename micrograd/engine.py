import numpy as np


class Value:
    """
    Wraps a scalar, vector, or matrix and tracks operations for automatic differentiation.

    The Value class is the core of the autograd engine. It stores data and its gradient,
    and builds a computational graph by tracking operations between Values.

    Example:
        >>> x = Value(2.0)
        >>> y = Value(3.0)
        >>> z = x * y + x
        >>> z.backward()  # Compute gradients
        >>> print(x.grad)  # dz/dx = y + 1 = 4.0
    """

    def __init__(self, data, _children=(), _op='', name=""):
        """
        Initialize a Value object.

        Args:
            data: The numerical data (scalar, list, or numpy array)
            _children: Tuple of parent Value objects (internal use for autograd)
            _op: String describing the operation that created this Value (internal)
            name: Optional name for debugging and visualization
        """
        # Convert any input to numpy array for consistent handling
        self.data = np.array(data, dtype=float)

        # Initialize gradient with same shape as data (starts at zero)
        self.grad = np.zeros_like(self.data)

        # Optional name for debugging
        self.name = name

        # Internal variables for building the computational graph
        self._backward = lambda: None  # Function to compute gradient
        self._prev = set(_children)     # Parent nodes in the graph
        self._op = _op                  # Operation that created this node

    def _handle_broadcast_gradient(self, gradient, target_shape):
        """
        Handle gradient broadcasting for backward pass.

        When forward pass uses broadcasting (e.g., adding a scalar to a matrix),
        we need to sum gradients appropriately in the backward pass.

        Args:
            gradient: The gradient flowing back
            target_shape: The shape we need to reduce to

        Returns:
            Gradient with the correct shape
        """
        if target_shape == gradient.shape:
            return gradient

        grad = gradient

        # Sum over extra dimensions that were broadcasted
        ndim_diff = len(gradient.shape) - len(target_shape)
        if ndim_diff > 0:
            axes_to_sum = tuple(range(ndim_diff))
            grad = np.sum(grad, axis=axes_to_sum)

        # Sum over dimensions that were size 1 in original (broadcasted to larger size)
        for i, (dim_grad, dim_target) in enumerate(zip(grad.shape, target_shape)):
            if dim_target == 1 and dim_grad > 1:
                grad = np.sum(grad, axis=i, keepdims=True)

        return grad

    def __add__(self, other):
        """
        Addition operation: supports Value + Value, Value + scalar, and broadcasting.

        Example:
            >>> a = Value([[1, 2], [3, 4]])
            >>> b = Value([[5, 6], [7, 8]])
            >>> c = a + b  # Element-wise addition
        """
        # Convert other to Value if it's a plain number
        other = other if isinstance(other, Value) else Value(other)

        # Forward pass: compute the sum
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            """
            Backward pass for addition: d(a+b)/da = 1, d(a+b)/db = 1

            The gradient flows equally to both inputs. If broadcasting occurred,
            we sum the gradients over the broadcasted dimensions.
            """
            # Handle broadcasting: reduce gradient to original shapes
            grad_self = self._handle_broadcast_gradient(out.grad, self.data.shape)
            grad_other = other._handle_broadcast_gradient(out.grad, other.data.shape)

            # Accumulate gradients
            self.grad += grad_self
            other.grad += grad_other

        out._backward = _backward
        return out

    def __mul__(self, other):
        """
        Multiplication operation: element-wise product with broadcasting support.

        Example:
            >>> a = Value(3.0)
            >>> b = Value(4.0)
            >>> c = a * b  # c.data = 12.0
        """
        # Convert other to Value if it's a plain number
        other = other if isinstance(other, Value) else Value(other)

        # Forward pass: element-wise multiplication
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            """
            Backward pass for multiplication: d(a*b)/da = b, d(a*b)/db = a

            Each input receives the gradient multiplied by the other input's value.
            """
            # Compute gradients using product rule
            grad_self = other.data * out.grad
            grad_other = self.data * out.grad

            # Handle broadcasting: reduce gradients to original shapes
            grad_self = self._handle_broadcast_gradient(grad_self, self.data.shape)
            grad_other = other._handle_broadcast_gradient(grad_other, other.data.shape)

            # Accumulate gradients
            self.grad += grad_self
            other.grad += grad_other

        out._backward = _backward
        return out

    def __matmul__(self, other):
        """
        Matrix multiplication: A @ B

        Example:
            >>> A = Value([[1, 2], [3, 4]])  # 2x2 matrix
            >>> B = Value([[5, 6], [7, 8]])  # 2x2 matrix
            >>> C = A @ B  # Matrix product
        """
        other = other if isinstance(other, Value) else Value(other)

        # Forward pass: matrix multiplication
        out = Value(self.data @ other.data, (self, other), '@',
                   name=f"{self.name}@{other.name}")

        def _backward():
            """
            Backward pass for matrix multiplication:
            - d(A@B)/dA = grad @ B.T
            - d(A@B)/dB = A.T @ grad

            This comes from the chain rule applied to matrix operations.
            """
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        """
        Power operation: raises Value to a scalar power.

        Example:
            >>> x = Value(3.0)
            >>> y = x ** 2  # y.data = 9.0
        """
        assert isinstance(other, (int, float)), "Only supporting int/float powers"

        # Forward pass: exponentiation
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            """
            Backward pass for power: d(x^n)/dx = n * x^(n-1)

            This is the standard power rule from calculus.
            """
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        """
        ReLU (Rectified Linear Unit) activation: max(0, x)

        ReLU is the most common activation function in deep learning.
        It outputs the input directly if positive, otherwise zero.

        Example:
            >>> x = Value([-1, 0, 1, 2])
            >>> y = x.relu()  # y.data = [0, 0, 1, 2]
        """
        # Forward pass: ReLU(x) = max(0, x)
        out = Value(np.maximum(0, self.data), (self,), 'ReLU', name=self.name)

        def _backward():
            """
            Backward pass for ReLU: d(ReLU(x))/dx = 1 if x > 0, else 0

            The gradient only flows through where the input was positive.
            """
            # Gradient is 1 where output > 0, otherwise 0
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        """
        Sigmoid activation: σ(x) = 1 / (1 + e^(-x))

        Squashes input to range (0, 1). Often used for binary classification.

        Example:
            >>> x = Value([0, 1, -1])
            >>> y = x.sigmoid()  # y.data ≈ [0.5, 0.73, 0.27]
        """
        # Forward pass: use numerically stable sigmoid computation
        # For x >= 0: σ(x) = 1 / (1 + e^(-x))
        # For x < 0:  σ(x) = e^x / (1 + e^x)  [avoids overflow]
        sigmoid_data = np.where(
            self.data >= 0,
            1 / (1 + np.exp(-self.data)),
            np.exp(self.data) / (1 + np.exp(self.data))
        )
        out = Value(sigmoid_data, (self,), 'sigmoid', name=self.name)

        def _backward():
            """
            Backward pass for sigmoid: dσ/dx = σ(x) * (1 - σ(x))

            This is a well-known derivative that uses the sigmoid output itself.
            """
            # Sigmoid derivative: σ'(x) = σ(x) * (1 - σ(x))
            sigmoid_grad = out.data * (1 - out.data)
            self.grad += sigmoid_grad * out.grad

        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        """
        Softmax activation: converts logits to probability distribution.

        Formula: softmax(x_i) = exp(x_i) / Σ exp(x_j)

        Used for multi-class classification. Outputs sum to 1 along the specified axis.

        Args:
            axis: The axis along which to apply softmax (default: -1, last axis)

        Example:
            >>> logits = Value([[1, 2, 3], [1, 2, 3]])  # shape (2, 3)
            >>> probs = logits.softmax(axis=-1)  # Each row sums to 1
        """
        x = self.data

        # Forward pass: numerically stable softmax
        # Subtract max for numerical stability (prevents overflow in exp)
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        y = exp_x / np.sum(exp_x, axis=axis, keepdims=True)

        out = Value(y, (self,), 'softmax')

        def _backward():
            """
            Backward pass for softmax:
            dL/dx = y * (dL/dy - Σ(dL/dy · y))

            Where:
              y = softmax output
              dL/dy = out.grad (gradient from next layer)

            This formula comes from the Jacobian of softmax.
            """
            # Ensure gradient array exists
            if self.grad is None:
                self.grad = np.zeros_like(self.data)

            dout = out.grad  # Gradient from next layer

            # Compute dot product along the axis
            dot = np.sum(dout * y, axis=axis, keepdims=True)

            # Apply softmax gradient formula
            dx = y * (dout - dot)

            # Accumulate gradient
            self.grad += dx

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        """
        Sum elements along an axis (or all elements).

        Args:
            axis: Axis to sum over (None means sum all elements)
            keepdims: Whether to keep the reduced dimension

        Example:
            >>> x = Value([[1, 2], [3, 4]])
            >>> total = x.sum()  # Scalar sum = 10
            >>> row_sums = x.sum(axis=1)  # [3, 7]
        """
        out = Value(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), 'sum')

        def _backward():
            """
            Backward pass for sum: d(sum(x))/dx = 1 for all elements

            The gradient is broadcast to match the input shape.
            """
            if axis is None:
                # Sum over all elements: gradient is same for all
                self.grad += np.full_like(self.data, out.grad)
            else:
                # Sum over specific axis: broadcast gradient back
                grad_shape = list(self.data.shape)
                if not keepdims:
                    # Need to restore the summed dimension
                    if isinstance(axis, int):
                        grad_shape[axis] = 1
                    else:
                        for ax in sorted(axis, reverse=True):
                            grad_shape[ax] = 1

                # Reshape and broadcast gradient to original shape
                grad_expanded = np.reshape(out.grad, grad_shape)
                self.grad += np.broadcast_to(grad_expanded, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        """
        Average of elements along an axis (or all elements).

        Args:
            axis: Axis to average over (None means all elements)
            keepdims: Whether to keep the reduced dimension

        Example:
            >>> x = Value([[1, 2, 3], [4, 5, 6]])
            >>> avg = x.mean()  # 3.5
            >>> col_avg = x.mean(axis=0)  # [2.5, 3.5, 4.5]
        """
        out = Value(np.mean(self.data, axis=axis, keepdims=keepdims), (self,), 'mean')

        def _backward():
            """
            Backward pass for mean: d(mean(x))/dx = 1/n for all elements

            Where n is the number of elements averaged over.
            """
            if axis is None:
                # Mean over all elements
                n_elements = self.data.size
                self.grad += np.full_like(self.data, out.grad / n_elements)
            else:
                # Mean over specific axis
                if isinstance(axis, int):
                    n_elements = self.data.shape[axis]
                else:
                    n_elements = np.prod([self.data.shape[ax] for ax in axis])

                # Prepare gradient shape
                grad_shape = list(self.data.shape)
                if not keepdims:
                    if isinstance(axis, int):
                        grad_shape[axis] = 1
                    else:
                        for ax in sorted(axis, reverse=True):
                            grad_shape[ax] = 1

                # Reshape, broadcast, and scale by 1/n
                grad_expanded = np.reshape(out.grad, grad_shape)
                self.grad += np.broadcast_to(grad_expanded, self.data.shape) / n_elements

        out._backward = _backward
        return out

    def mse(self, target):
        """
        Mean Squared Error loss: MSE = 1/(2n) * sum((target - prediction)²)

        The factor 1/2 simplifies the gradient computation.
        """
        target = target if isinstance(target, Value) else Value(target)

        # Forward pass: MSE = 1/(2n) * sum((target - self)²)
        diff = target - self
        squared_diff = diff ** 2
        n = np.prod(self.data.shape) if hasattr(self.data, 'shape') else 1
        out = Value(np.sum(squared_diff.data) / (2 * n), (self, target), 'mse')

        def _backward():
            """
            Backward pass for MSE with 1/2 factor:
            d(MSE)/d(prediction) = -1/n * (target - prediction) = 1/n * (prediction - target)
            """
            diff_data = target.data - self.data
            grad = (-1.0 / n) * diff_data

            self.grad += grad * out.grad
            target.grad += (-grad) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """
        Perform backpropagation: compute gradients for all Values in the graph.

        This method implements automatic differentiation using reverse-mode
        accumulation (backpropagation). It traverses the computational graph
        in reverse topological order and applies the chain rule.

        Must be called on a scalar Value (the loss).

        Example:
            >>> x = Value(2.0)
            >>> y = x * 3 + 1
            >>> y.backward()
            >>> print(x.grad)  # dy/dx = 3.0
        """
        # Build topological order: children before parents
        topo = []
        visited = set()

        def build_topo(v):
            """Recursively build topological sort of the computational graph."""
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Initialize gradient of output to 1 (dL/dL = 1)
        self.grad = np.ones_like(self.data)

        # Traverse graph in reverse: apply chain rule to compute all gradients
        for v in reversed(topo):
            v._backward()

    # Reverse and derived operations (use the basic operations defined above)

    def __neg__(self):
        """Negation: -x = -1 * x"""
        return self * -1

    def __radd__(self, other):
        """Right addition: other + self (when other is not a Value)"""
        return self + other

    def __sub__(self, other):
        """Subtraction: a - b = a + (-b)"""
        return self + (-other)

    def __rsub__(self, other):
        """Right subtraction: other - self"""
        return other + (-self)

    def __rmul__(self, other):
        """Right multiplication: other * self (when other is not a Value)"""
        return self * other

    def __truediv__(self, other):
        """Division: a / b = a * b^(-1)"""
        return self * other**-1

    def __rtruediv__(self, other):
        """Right division: other / self"""
        return other * self**-1

    @property
    def T(self):
        """
        Transpose of the Value (like numpy's .T).

        Preserves the computational graph for backpropagation.

        Example:
            >>> A = Value([[1, 2], [3, 4]])  # 2x2 matrix
            >>> A_T = A.T  # Transposed: [[1, 3], [2, 4]]
        """
        out = Value(self.data.T, (self,), 'T', name=f"{self.name}.T")

        def _backward():
            """Backward pass for transpose: gradient is also transposed"""
            self.grad += out.grad.T

        out._backward = _backward
        return out

    def __repr__(self):
        """Return a readable string representation of the Value."""
        name_str = f"'{self.name}' " if self.name else ""
        op_str = f" from {self._op}" if self._op else ""
        return f"Value({name_str}data={self.data}, grad={self.grad}{op_str})"
