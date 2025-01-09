# PyTorch Extended Autograd

## Introduction

**torch_xa** is a Python-based implementation of an extended autograd engine that emulates PyTorch’s computational graph mechanism while providing **arbitrary-order partial derivatives**. Unlike PyTorch’s native autograd, which stops at first-order derivatives, **torch_xa** can propagate higher-order derivatives throughout the computational graph for more advanced gradient-based computations.

### Key Features

- **Arbitrary-Order Partial Derivatives**: Extends beyond first-order derivatives, supporting higher-order derivative calculations within the same computational graph.
- **PyTorch Integration**: Operates alongside PyTorch tensors and modules, retaining familiar behaviors (e.g., `requires_grad` flags) while introducing additional functionality.
- **Unified API**: Provides interfaces (`backward` function and `Superset` class) that mirror PyTorch’s usage patterns, minimizing the learning curve.
- **Python 3.12+**: Developed and tested on Python 3.12, ensuring compatibility with the latest Python enhancements and optimizations.

**Note**: The core functionality relies solely on PyTorch, although the repository includes `pytest` in its requirements for development and testing purposes. The extended autograd engine itself does not require `pytest` to function.

---

## Installation

To install **torch_xa** and start exploring higher-order derivatives, follow these steps:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/mntsx/torch_xa.git
   cd torch_xa
   ```

2. **Create a Virtual Environment**

   Using a virtual environment is recommended to manage dependencies and maintain a clean environment.

   - **Windows**:
     ```bash
     py -3.12 -m venv .venv
     ```
   - **macOS/Linux**:
     ```bash
     python3.12 -m venv .venv
     ```

3. **Activate the Virtual Environment**

   - **Windows**:
     ```bash
     .venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```

4. **Install Dependencies**

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
   
   **Note**: Only PyTorch is strictly required to run **torch_xa**. The `pytest` package in `requirements.txt` is used solely for testing purposes.

---

## User Guide

The **torch_xa** package offers two primary approaches for computing higher-order partial derivatives:

1. **`torch_xa.backward` Function**  
   This function behaves similarly to PyTorch’s native `torch.Tensor.backward`, but it supports partial derivatives of any order. Upon calling `torch_xa.backward`, **all leaf tensors** in the computational graph (`torch.Tensor.is_leaf = True`) will receive a new attribute `torch.Tensor.partials`. Only leaf tensors that have `requires_grad=True` retain these computed partials; all others will keep `partials=None`.

   **Function Arguments**:
   - **source** *(torch.Tensor)*: The tensor from which partial derivatives are computed.
   - **order** *(int)*: The order of the derivative to compute.
   - **target** *(torch.Tensor, optional)*: If specified, partials are computed only for the path(s) in the graph leading to `target`.
   - **configurations** *(list, optional)*: A list of configuration classes that can modify or extend the default backward process.

   **Key Differences from `torch.Tensor.backward`**:
   1. **Non-Scalar Source**: Extended autograd supports partial derivatives from any tensor (not necessarily scalar).
   2. **Dimensional Flattening**: Because extended autograd handles both non-scalar sources and higher-order derivatives, the dimensionality related to “intravariable” space is flattened to simplify the identification of each derivative component.

   **Example Usage**:
   ```python
   import torch
   from torch_xa import backward

   T0 = torch.rand(size=(4, 6), requires_grad=True)
   T1 = torch.rand(size=(6, 8), requires_grad=True)
   O = torch.mm(T0, T1)  # Some operation in the graph

   # Compute second-order partial derivatives
   backward(source=O, order=2)

   assert T0.partials[0].shape == (O.numel(), T0.numel())
   assert T1.partials[0].shape == (O.numel(), T1.numel())

   # Optionally, capture the Superset for further graph management
   superset = backward(source=O, order=2)
   ```

2. **`torch_xa.Superset` Object**  
   The `Superset` class provides additional tools for handling the computational graph. It includes a method `Superset.backward` that offers functionality equivalent to `torch_xa.backward`. Moreover, `Superset` allows you to:
   - Remove the `partials` attribute from modified tensors.
   - (De)activate gradient retention for intermediate tensors.
   - Access non-leaf tensors’ partials via `Superset.operator_partials`, since non-leaf tensors cannot natively store partials as attributes in PyTorch.

   **Example Usage**:
   ```python
   import torch
   from torch_xa import Superset

   T0 = torch.rand(size=(4, 6), requires_grad=True)
   T1 = torch.rand(size=(6, 8), requires_grad=True)
   O = torch.mm(T0, T1)

   superset = Superset.construct(source=O)
   superset.backward(order=2)

   assert T0.partials[0].shape == (O.numel(), T0.numel())
   assert T1.partials[0].shape == (O.numel(), T1.numel())
   ```

---

## Testing

This repository includes a suite of tests to verify the correctness of various components within **torch_xa**. To run the tests, execute:

```bash
pytest .
```

These tests validate that different modules (e.g., graph management, partial derivative calculation, configuration handling) are working correctly.

---

## Benchmarking

The `benchmarks` directory contains scripts that measure execution time and efficiency of certain parts of the extended autograd engine. These benchmarks can be useful for assessing performance in a variety of scenarios, such as higher-order derivative computations on large tensors.

To run benchmarks, navigate to the `benchmarks` directory and execute the relevant script(s). For example:

```bash
python -m benchmarks.some_benchmark_script
```

Refer to individual scripts within the `benchmarks/` folder for specific instructions or required parameters.

---

## Repository Structure

Below is an overview of the **torch_xa** repository’s structure, illustrating how source code, tests, and benchmark scripts are organized. Each directory focuses on a distinct aspect of the extended autograd engine.

```
.
├── benchmarks
│   └── utils
├── src
│   ├── autograd
│   │   ├── configurations
│   │   │   ├── exchangers.py
│   │   │   ├── flags.py
│   │   │   ├── selectors.py
│   │   │   └── __init__.py
│   │   ├── engine
│   │   │   ├── backprop
│   │   │   │   ├── contraction.py
│   │   │   │   └── derivation.py
│   │   │   ├── graph.py
│   │   │   ├── interfaces.py
│   │   │   ├── scheduler.py
│   │   │   └── __init__.py
│   │   ├── XAF
│   │   │   ├── activations
│   │   │   │   ├── relu.py
│   │   │   │   ├── sigmoid.py
│   │   │   │   ├── softmax.py
│   │   │   │   ├── tanh.py
│   │   │   │   └── __init__.py
│   │   │   ├── attention
│   │   │   │   └── __init__.py
│   │   │   ├── convolutional
│   │   │   │   └── __init__.py
│   │   │   ├── indexation
│   │   │   │   └── __init__.py
│   │   │   ├── linalg
│   │   │   │   ├── addmm.py
│   │   │   │   ├── mm.py
│   │   │   │   └── __init__.py
│   │   │   ├── normalization
│   │   │   │   └── __init__.py
│   │   │   └── reshape
│   │   │       ├── permute.py
│   │   │       ├── t.py
│   │   │       ├── transpose.py
│   │   │       ├── view.py
│   │   │       └── __init__.py
│   │   ├── __init__.py
│   │   └── XAF
│   │       ├── accumulation.py
│   │       ├── base.py
│   │       ├── test.py
│   │       └── __init__.py
│   ├── tensors
│   │   ├── functional.py
│   │   ├── objects.py
│   │   └── __init__.py
│   └── utils
│       ├── partials.py
│       ├── relationships.py
│       ├── types.py
│       └── __init__.py
└── tests
    └── utils
```

### Directory and File Highlights

- **`src/autograd/engine/`**  
  Manages the computational graph.  
  - **`graph.py`**: Implements the `Graph` and `Node` classes.  
  - **`scheduler.py`**: An auxiliary scheduler that manages node transition in a cascading fashion.  
  - **`interfaces.py`**: Contains the `backward` function and `Superset` class, which are the primary user-facing APIs.  
  - **`backprop/`**: Houses lower-level logic for partial derivative backpropagation.

- **`src/autograd/configurations/`**  
  Contains additional configurations that can be passed to `backward`.  
  - **`selectors.py`**: Classes for selecting different backward functions for various modules.  
  - **`exchangers.py`**: Classes that modify selectors partially.  
  - **`flags.py`**: Classes used as execution markers.

- **`src/tensors/`**  
  Deals with tensor customization and special tensor objects.  
  - **`functional.py`**: Utility functions for creating or modifying tensors.  
  - **`objects.py`**: Defines classes for specialized tensor types.

- **`src/utils/`**  
  Contains helper functions used across the repository.  
  - **`partials.py`**: Functions for manipulating `Partial` objects.  
  - **`relationships.py`**: Facilitates links between forward modules and backward modules.  
  - **`types.py`**: Internal type definitions.

- **`benchmarks/`**  
  Benchmark programs measuring performance and execution time of key extended autograd components.

- **`tests/`**  
  Test scripts using `pytest` to ensure the reliability and correctness of the extended autograd engine.

---

Thank you for using **torch_xa**! By enabling higher-order derivative computations in a familiar PyTorch-like environment, **torch_xa** aims to simplify research and development in advanced gradient-based methods. If you have questions or wish to contribute, please feel free to submit issues or pull requests. Enjoy exploring higher-order autograd!
