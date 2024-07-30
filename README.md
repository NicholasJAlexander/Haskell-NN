<p align="center">
  <img src="https://cdn-icons-png.flaticon.com/512/6295/6295417.png" width="100" />
</p>
<p align="center">
    <h1 align="center">HASKELL-NN</h1>
</p>
<p align="center">
    <em>HTTP error 401 for prompt `slogan`</em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/NicholasJAlexander/Haskell-NN?style=flat&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/NicholasJAlexander/Haskell-NN?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/NicholasJAlexander/Haskell-NN?style=flat&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/NicholasJAlexander/Haskell-NN?style=flat&color=0080ff" alt="repo-language-count">
<p>
<p align="center">
		<em>Developed with the software and tools below.</em>
</p>
<p align="center">
	<img src="https://img.shields.io/badge/GNU%20Bash-4EAA25.svg?style=flat&logo=GNU-Bash&logoColor=white" alt="GNU%20Bash">
	<img src="https://img.shields.io/badge/Jupyter-F37626.svg?style=flat&logo=Jupyter&logoColor=white" alt="Jupyter">
	<img src="https://img.shields.io/badge/Haskell-5D4F85.svg?style=flat&logo=Haskell&logoColor=white" alt="Haskell">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
</p>
<hr>

##  Quick Links

> - [ Overview](#-overview)
> - [ Features](#-features)
> - [ Repository Structure](#-repository-structure)
> - [ Modules](#-modules)
> - [ Getting Started](#-getting-started)
>   - [ Installation](#-installation)
>   - [ Running Haskell-NN](#-running-Haskell-NN)
>   - [ Tests](#-tests)
> - [ Project Roadmap](#-project-roadmap)
> - [ Contributing](#-contributing)
> - [ License](#-license)
> - [ Acknowledgments](#-acknowledgments)

---

##  Overview

This project presents a Haskell-based Neural Network (NN) library, designed to leverage Haskell's functional programming paradigm for machine learning tasks. The library emphasizes mathematical fidelity, type safety, and functional purity, providing a framework for NN implementation. Key features include pure functions, strong static typing, and higher-order functions, ensuring predictable and maintainable code. The implementation covers core NN functionalities such as forward and backward propagation, weight updates, and training processes, all while closely aligning with mathematical representations found in academic literature.

This library serves as a tool for researchers and developers interested in exploring the intersection of functional programming and machine learning. By maintaining a pure Haskell implementation, it avoids external dependencies and provides a highly type-safe environment, catching errors at compile time and reducing runtime issues. Despite some challenges in performance optimization for large datasets, the library demonstrates competitive results on smaller datasets and offers a solid foundation for future enhancements and optimizations. The project's code is available on GitHub, inviting contributions and further exploration in advancing functional approaches to neural network development.
---


##  Repository Structure

```sh
└── Haskell-NN/
    ├── NeuralNetwork.cabal
    ├── benchmarks
    │   ├── benchmark.hs
    │   ├── benchmark.py
    │   ├── benchmark_depth.sh
    │   ├── benchmark_epoch_batches.sh
    │   └── test_stats
    │       ├── depth_neurons.csv
    │       ├── depth_neurons.txt
    │       ├── epochs_batch.csv
    │       ├── epochs_batch.txt
    │       └── stats_to_csv.ipynb
    ├── cabal.project.local
    ├── examples
    │   ├── .DS_Store
    │   ├── IRIS
    │   │   ├── iris.csv
    │   │   ├── iris.hs
    │   │   ├── iris.py
    │   │   └── iris_results.txt
    │   └── MNIST
    │       ├── .DS_Store
    │       ├── mnist.hs
    │       ├── mnist.py
    │       ├── mnist_results.txt
    │       ├── t10k-images-idx3-ubyte
    │       ├── t10k-labels-idx1-ubyte
    │       ├── train-images-idx3-ubyte
    │       └── train-labels-idx1-ubyte
    ├── mnist.prof
    ├── run_iris.sh
    ├── run_mnist.sh
    ├── src
    │   ├── ActivationFunctions.hs
    │   ├── Matrix.hs
    │   └── NN.hs
    └── test
        └── NNTests.hs
```

---

## Library Modules

The `NeuralNetwork` library is composed of several Haskell modules, each responsible for different aspects of the neural network implementation. Below is an overview of the primary modules:

### NN.hs
This module is the core of the neural network implementation. It includes the data structures and functions necessary to construct, train, and evaluate neural networks. Key functionalities provided by this module include:
- **Network Definition:** Defines the `Layer` and `BackpropNet` data types that represent the structure of the neural network.
- **Forward Propagation:** Implements the `propagate` function to process inputs through the network layers.
- **Backward Propagation:** Implements the `backpropagate` function to compute the gradient of the loss function with respect to network weights.
- **Training:** Provides functions for training the network using gradient descent, such as `trainBatch` and `trainEpochs`.

### Matrix.hs
This module handles all matrix operations, which are fundamental to the neural network computations. It includes:
- **Matrix and Vector Definitions:** Defines data structures for matrices and vectors, crucial for representing neural network weights and activations.
- **Matrix Operations:** Implements various matrix operations like addition, multiplication, and transposition that are essential for forward and backward propagation.
- **Optimizations:** Includes optimized functions to enhance the performance of matrix operations, ensuring efficient computation during network training and evaluation.

### ActivationFunctions.hs
This module encapsulates the activation functions used within the neural network layers. Activation functions introduce non-linearity into the network, enabling it to learn complex patterns. The module provides:
- **Activation Function Definitions:** Defines various activation functions such as `sigmoid`, `tanh`, and `relu`, along with their derivatives.
- **Function Application:** Implements functions to apply activation functions to vectors and matrices, ensuring seamless integration with the rest of the network operations.
- **Extensibility:** Allows for easy addition of new activation functions, facilitating experimentation with different neural network architectures.

These modules collectively form the foundation of the `NeuralNetwork` library, providing a robust and flexible framework for developing and experimenting with neural networks in Haskell. Each module is designed to be modular and extensible, allowing users to customize and extend the library to meet their specific needs.



## Getting Started

To get started with the `NeuralNetwork` library, you'll need a basic understanding of Haskell and the `cabal` build system. Follow the steps below to set up the library and run the examples provided.

### Prerequisites

Ensure you have the following installed on your system:
- [GHC (Glasgow Haskell Compiler)](https://www.haskell.org/ghc/)
- [Cabal](https://www.haskell.org/cabal/)

You can install GHC and Cabal using the [Haskell Platform](https://www.haskell.org/platform/) or through your system's package manager.

### Installation

1. **Clone the Repository:**
   Clone the repository from GitHub to your local machine.
   ```bash
   git clone https://github.com/NicholasJAlexander/Haskell-NN.git
   cd Haskell-NN
   ```

2. **Set Up the Project:**
   Initialize the project using `cabal`. This will install the necessary dependencies as specified in the `cabal` file.
   ```bash
   cabal update
   cabal build
   ```

3. **Run the Examples:**
   The repository includes examples for training and evaluating neural networks on the MNIST and IRIS datasets. To run these examples, use the following commands:

   - **Run the MNIST Example:**
     ```bash
     cabal run mnist
     ```

   - **Run the IRIS Example:**
     ```bash
     cabal run iris
     ```

4. **Run the Benchmarks:**
   To benchmark the performance of the library, use the following command:
   ```bash
   cabal run benchmark
   ```

5. **Run the Tests:**
   The project includes some property testing ensure the library functions correctly. To run the tests, use:
   ```bash
   cabal test
   ```

### Project Structure

The project is structured as follows:

- **src/**: Contains the source code for the library.
  - `ActivationFunctions.hs`: Defines activation functions and their derivatives.
  - `Matrix.hs`: Implements matrix and vector operations.
  - `NN.hs`: Core neural network implementation including data structures and training functions.

- **examples/**: Contains example applications using the library.
  - **MNIST/**: Example for training and evaluating a neural network on the MNIST dataset.
  - **IRIS/**: Example for training and evaluating a neural network on the IRIS dataset.

- **benchmarks/**: Contains benchmarking code to evaluate the library's performance.
  - `benchmark.hs`: Benchmarking script.

- **test/**: Contains the test suite for the library.
  - `NNTests.hs`: Tests for the neural network implementation.

### Additional Notes

- **GHC Options:**
  The `ghc-options` field in the `cabal` file includes `-Wall -fprof-auto -rtsopts` to enable all warnings, automatic profiling, and runtime system options. These settings help in debugging and performance analysis.

- **Documentation:**
  Ensure to check the [Haddock](https://www.haskell.org/haddock/) documentation generated from the source code for detailed information about the functions and types provided by the library.

By following these steps, you should have the `NeuralNetwork` library set up and ready to use. Explore the examples and modify them to suit your needs, or start building your own neural network applications using Haskell!


