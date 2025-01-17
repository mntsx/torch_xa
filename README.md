# PyTorch Extended Autograd

<br>

## Introduction

**torch_xa** is a Python-based implementation of an extended autograd engine that emulates PyTorch’s computational graph mechanism while providing **arbitrary-order partial derivatives**. Unlike PyTorch’s native autograd, which stops at first-order derivatives, **torch_xa** can propagate higher-order derivatives throughout the computational graph for more advanced gradient-based computations.

### Key Features

- **Arbitrary-Order Partial Derivatives**: Extends beyond first-order derivatives, supporting higher-order derivative calculations within the same computational graph.
- **PyTorch Integration**: Operates alongside PyTorch tensors and modules, retaining familiar behaviors (e.g., `requires_grad` flags) while introducing additional functionality.
- **Unified API**: Provides interfaces (`backward` function and `Superset` class) that mirror PyTorch’s usage patterns, minimizing the learning curve.
- **Python 3.12+**: Developed and tested on Python 3.12, ensuring compatibility with the latest Python enhancements and optimizations.

**Note**: The core functionality relies solely on PyTorch, although the repository includes `pytest` in its requirements for development and testing purposes. The extended autograd engine itself does not require `pytest` to function.

---

<br>

## Installation

To install **torch_xa** and start exploring higher-order derivatives, follow these steps:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/mntsx/torch_xa.git
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
   cd torch_xa
   pip install -r requirements.txt
   ```
---

<br>

## User Guide

**Key Differences** between torch native autograd and torch_xa extended autograd:

1. **Multiple First-Order Partial Derivatives**:  
   **torch_xa** allows users to compute the N first-order partial derivatives of the output tensor with respect to each input tensor. These derivatives are returned as a `Tuple[Tensor, ...]`, enabling simultaneous access to multiple gradient directions and facilitating more complex gradient-based computations.
2. **Non-Scalar Source**:  
   Extended autograd supports partial derivatives from any tensor (not necessarily scalar).
3. **Dimensional Flattening**:  
   The gradients calculated by PyTorch's autograd from a scalar only consider the dimensionality of the first partial derivative's conforming dual spaces, maintaining this "intravariable" dimensionality. In contrast, Extended Autograd supports partial derivatives from non-scalar tensors and higher-order derivatives, which requires managing both the output tensor's dimensionality and the "intervariable" dimensions (multiple derivations with respect to the same variable). To simplify derivative identification for users, the "intravariable" dimensionality is flattened.  
   **Note**: This flattening does not hinder usability, as the original dimensionality can be easily recovered when users have access to the input and output tensors.

The **torch_xa** package offers two primary approaches for launching the extended autograd:

1. **`torch_xa.backward` Function**  
   This function behaves similarly to PyTorch’s native `torch.Tensor.backward`, but it supports partial derivatives of any order. Upon calling `torch_xa.backward`, **all leaf tensors** in the computational graph (`torch.Tensor.is_leaf = True`) will receive a new attribute `torch.Tensor.xgrad`. Only leaf tensors that have `requires_grad=True` retain these computed partials; all others will keep `partials=None`.

   **Function Arguments**:
   - **source** *(torch.Tensor)*: The tensor from which partial derivatives are computed.
   - **order** *(int)*: The order of the derivative to compute.
   - **target** *(torch.Tensor, optional)*: If specified, partials are computed only for the path(s) in the graph leading to `target`.
   - **configurations** *(list, optional)*: A list of configuration classes that can modify or extend the default backward process.

   **Example Usage**:

   ```python
   import torch
   from torch_xa import backward

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T0 = torch.rand(size=(8,), requires_grad=True, device=device)
    T1 = torch.rand(size=(4, 6), requires_grad=True, device=device)
    T2 = torch.rand(size=(6, 8), requires_grad=True, device=device)

    O0 = torch.addmm(input=T0, mat1=T1, mat2=T2)
    O1 = torch.relu(O0)
    O2 = torch.add(O1, T0)
    O3 = torch.softmax(O2, dim=1)

    order = 3
    backward(source=O3, order=order)

    for i in range(order):
        assert T0.xgrad[i].shape == (O3.numel(), *((i + 1) * (T0.numel(),) ))
        assert T1.xgrad[i].shape == (O3.numel(), *((i + 1) * (T1.numel(),) ))
        assert T2.xgrad[i].shape == (O3.numel(), *((i + 1) * (T2.numel(),) ))
   ```

2. **`torch_xa.Superset` Object**  
   The `Superset` class provides additional tools for handling the computational graph. It includes a method `Superset.backward` that offers functionality equivalent to `torch_xa.backward`. Moreover, `Superset` allows you to:
   - Remove the `xgrad` attribute from modified tensors.
   - (De)Activate gradient retention for intermediate tensors.
   - Access the partial derivatives of non-leaf tensors through `Superset.operator_partials`, since non-leaf tensors cannot be natively accessed within the PyTorch computational graph and, consequently, cannot have the `xgrad` attribute directly assigned to them.

   **Example Usage**:

   ```python
   import torch
   from torch_xa import Superset

    # ...

    order = 3
    superset = Superset.construct(source=O3)
    superset.backward(order=order)
   
    # ...
   ```

 ---

 <br>

## Testing

This repository includes a suite of tests to verify the correctness of various components within **torch_xa**. To run the tests, navigate to the `tests` directory and execute:

```bash
pytest .
```

These tests validate that different modules (e.g., graph management, partial derivative calculation, configuration handling) are working correctly.

---

<br>

## Benchmarking

The `benchmarks` directory contains scripts that measure execution time and efficiency of certain parts of the extended autograd engine. These benchmarks can be useful for assessing performance in a variety of scenarios, such as higher-order derivative computations on large tensors.

To run benchmarks, navigate to the `benchmarks` directory and execute the relevant script(s). For example:

```bash
python -m benchmarks.some_benchmark_script
```

Refer to individual scripts within the `benchmarks/` folder for specific instructions or required parameters.

---

<br>

## Compatibility

The proper functioning of extended autograd with each different PyTorch operator requires the implementation of its corresponding extended backward function. For this reason, extended autograd is only compatible with a subset of PyTorch operators. Below is a list of these operators.

### Activations

- **ReLU**: `torch.relu`, `torch.nn.ReLU`, `torch.nn.functional.relu`
- **Sigmoid**: `torch.sigmoid`, `torch.nn.Sigmoid`, `torch.nn.functional.sigmoid`
- **Softmax**: `torch.softmax`, `torch.nn.Softmax`, `torch.nn.functional.softmax`
- **Tanh**: `torch.tanh`, `torch.nn.Tanh`, `torch.nn.functional.tanh`

### Matrix Multiplication

- **addmm**: `torch.addmm`, `torch.nn.Linear`, `torch.nn.functional.linear`
- **mm**: `torch.mm`, `torch.matmul`, `torch.nn.Linear`, `torch.nn.functional.linear`

### Products

- **Multiply**: `*`, `torch.mul`, `torch.multiply`, `torch.Tensor.mul`, `torch.nn.functional.dropout`
- **Product**: `torch.prod`, `torch.Tensor.prod`

### Reshape

- **Permute**: `torch.permute`, `torch.Tensor.permute`, `torch.Tensor.T`
- **T**: `torch.t`, `torch.Tensor.t`
- **Transpose**: `torch.transpose`, `torch.Tensor.transpose`
- **View**: `torch.view`, `torch.Tensor.view`, `torch.flatten`, `torch.Tensor.flatten`

### Summations

- **Add**: `+`, `torch.add`, `torch.Tensor.add`
- **Sum**: `torch.sum`, `torch.Tensor.sum`

---

<br>

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
│   │   │   └── selectors.py
│   │   ├── engine
│   │   │   ├── symbolic
│   │   │   │   ├── derivation.py
│   │   │   │   └── polinomial.py
│   │   │   ├── backprop.py
│   │   │   ├── graph.py
│   │   │   ├── interfaces.py
│   │   │   └── scheduler.py
│   │   └── XAF
│   │       ├── activations
│   │       ├── attention
│   │       ├── convolutional
│   │       ├── indexation
│   │       ├── linalg
│   │       ├── normalization
│   │       ├── product
│   │       ├── reshape
│   │       ├── summation
│   │       ├── accumulation.py
│   │       ├── base.py
│   │       └── test.py
│   ├── tensors
│   │   ├── functional.py
│   │   └── objects.py
│   └── utils
│       ├── partials.py
│       ├── relationships.py
│       └── types.py
└── tests
    └── utils
```

### Directory and File Highlights

- **`src/autograd/`**  
  Contains the core implementation of the extended autograd engine.

  - **`engine/`**  
    Manages the computational graph.
    - **`backprop.py`**: It contains the `contract` and `hadamard` functions, which define the two methods for performing the necessary contractions to carry out the backpropagation of partial derivatives.
    - **`graph.py`**: Implements the `Graph` and `Node` classes, which represent the computational graph and its constituent nodes.
    - **`scheduler.py`**: An auxiliary scheduler that manages node transitions in a cascading fashion, ensuring proper order of operations during backpropagation.
    - **`interfaces.py`**: Contains the `backward` function and `Superset` class, which are the primary user-facing APIs for computing partial derivatives and managing the computational graph.
    - **`symbolic/`**: Includes utility functions and classes to obtain the necessary expressions for calculating the partial derivatives of some PyTorch operators.
      - **`derivation.py`**: Contains classes used to obtain expressions of partial derivatives of a layer with respect to its input through the preceding layer, facilitating higher-order derivative calculations.
      - **`polinomial.py`**: Contains classes used to obtain the polynomial expressions necessary for calculating the derivatives of certain operators such as the sigmoid or hyperbolic tangent.

  - **`configurations/`**  
    Contains additional configurations that can be passed to `backward`.
    - **`selectors.py`**: Classes for selecting different backward functions for various modules, allowing customization of the backward pass behavior.
    - **`exchangers.py`**: Classes that modify selectors partially, enabling dynamic adjustments to the backward function selection process.
    - **`flags.py`**: Classes used as execution markers, signaling specific actions or states within the backward computation process.

  - **`XAF/`**  
    Contains the definition of the XAF (Extended Autograd Functions) classes corresponding to various modules.

- **`src/tensors/`**  
  Deals with tensor customization and special tensor objects.
  - **`functional.py`**: Utility functions for creating or modifying tensors, enabling specialized tensor operations required by the extended autograd engine.
  - **`objects.py`**: Defines classes for specialized tensor types, such as custom tensor subclasses that integrate seamlessly with the extended autograd functionalities.

- **`src/utils/`**  
  Contains helper functions used across the repository.
  - **`partials.py`**: Functions for manipulating `Partial` objects, which represent partial derivatives within the computational graph.
  - **`relationships.py`**: Facilitates links between forward modules and backward modules, ensuring coherent flow of derivative computations.
  - **`types.py`**: Internal type definitions that enhance code readability and maintainability through improved type hinting and structure.
