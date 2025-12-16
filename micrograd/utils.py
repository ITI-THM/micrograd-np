"""
Visualization utilities for micrograd computational graphs.

This module provides functions to visualize the computational graph created by
Value objects, showing the flow of data and gradients through operations.
"""

import numpy as np
from graphviz import Digraph


def trace(root):
    """
    Trace the computational graph starting from a root Value node.

    Performs a depth-first traversal of the computational graph to collect
    all nodes and edges. This is used internally by draw_dot() to build
    the visualization.

    Args:
        root: A Value object representing the output of a computation

    Returns:
        tuple: (nodes, edges) where:
            - nodes: set of all Value objects in the graph
            - edges: set of (parent, child) tuples representing connections

    Example:
        >>> from micrograd.engine import Value
        >>> x = Value(2.0)
        >>> y = Value(3.0)
        >>> z = x * y + x
        >>> nodes, edges = trace(z)
        >>> len(nodes)  # Should have nodes for x, y, x*y, and z
        4
    """
    nodes, edges = set(), set()

    def build(v):
        """Recursively add node and its predecessors to the graph."""
        if v not in nodes:
            nodes.add(v)
            # For each parent (predecessor) of this node, add edge and recurse
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges

def _short_repr(arr, max_shape=(4, 4)):
    """
    Create a compact string representation of numpy arrays for visualization.

    Formats arrays intelligently based on their size:
    - Scalars: Single value with 4 decimal places
    - Small arrays/matrices: Full pretty-printed format
    - Large arrays: Shape and sample values (first 3 and last 3)

    Args:
        arr: Numpy array or array-like to format
        max_shape: Tuple (max_rows, max_cols) for full printing

    Returns:
        str: Formatted string representation

    Example:
        >>> _short_repr(np.array(3.14159))
        '3.1416'
        >>> _short_repr(np.array([1, 2, 3]))
        '\\n1.0000 2.0000 3.0000'
        >>> _short_repr(np.random.randn(100, 100))
        'shape=(100, 100)\\n[0.1234, -0.5678, ..., 0.9012]'
    """
    arr = np.array(arr)

    # Handle scalar values
    if arr.ndim == 0:
        return f"{arr.item():.4f}"

    # Handle empty arrays
    if arr.size == 0:
        return "[]"

    # Pretty-print small arrays/matrices
    if arr.ndim <= 2 and arr.shape[0] <= max_shape[0] and arr.shape[-1] <= max_shape[1]:
        lines = []
        for row in arr:
            if arr.ndim == 1:
                # 1D array: single line
                lines.append(" ".join(f"{x:.4f}" for x in arr))
                break
            # 2D array: one line per row
            lines.append(" ".join(f"{x:.4f}" for x in row))
        return "\\n" + "\\n".join(lines)
    else:
        # Summarize large arrays with shape and sample values
        shape = arr.shape
        flat = arr.flatten()
        vals = ", ".join(f"{x:.4f}" for x in flat[:3]) + ", ..., " + ", ".join(f"{x:.4f}" for x in flat[-3:])
        return f"shape={shape}\\n[{vals}]"

def draw_dot(root, format='svg', rankdir='LR'):
    """
    Visualize the computational graph of a Value object as a directed graph.

    Creates a Graphviz diagram showing:
    - Value nodes with their data and gradients
    - Operation nodes (+, *, relu, etc.)
    - Edges showing data flow through the computation

    This is extremely useful for understanding and debugging neural networks,
    as it shows exactly how gradients flow backward through operations.

    Args:
        root: A Value object (typically the loss) to visualize from
        format: Output format ('svg', 'png', 'pdf', etc.)
        rankdir: Graph direction - 'LR' (left-right) or 'TB' (top-bottom)

    Returns:
        Digraph: A graphviz Digraph object that can be rendered or displayed

    Example:
        >>> from micrograd.engine import Value
        >>> from micrograd.utils import draw_dot
        >>> x = Value(2.0, name='x')
        >>> y = Value(-3.0, name='y')
        >>> z = x * y
        >>> z.name = 'z'
        >>> z.backward()
        >>> graph = draw_dot(z)
        >>> graph.render('computation_graph')  # Saves as SVG
        >>> # In Jupyter: display(graph)

    Note:
        Requires graphviz to be installed:
        - Python: pip install graphviz
        - System: apt install graphviz (Ubuntu) or brew install graphviz (Mac)
    """
    assert rankdir in ['LR', 'TB'], "rankdir must be 'LR' (left-right) or 'TB' (top-bottom)"

    # Trace the computational graph to get all nodes and edges
    nodes, edges = trace(root)

    # Create a new directed graph with specified layout direction
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})

    # Add nodes to the graph
    for n in nodes:
        # Format data and gradient for display
        data_str = _short_repr(n.data)
        grad_str = _short_repr(n.grad)

        # Create label showing name, data, and gradient
        label = f'{{ {n.name} | {{ data {data_str} | grad {grad_str} }} }}'

        # Add the Value node as a record-shaped box
        dot.node(name=str(id(n)), label=label, shape='record')

        # If this node was created by an operation, add an operation node
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            # Connect operation to its result
            dot.edge(str(id(n)) + n._op, str(id(n)))

    # Add edges showing data flow between nodes
    for n1, n2 in edges:
        # Connect parent (n1) to operation that created child (n2)
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
