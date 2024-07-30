#!/bin/bash
cabal build mnist
echo "Running mnist.hs" >> ./examples/MNIST/mnist_results.txt 2>&1
gtime -v cabal run mnist >> ./examples/MNIST/mnist_results.txt 2>&1
echo "Running mnist.py" >> ./examples/MNIST/mnist_results.txt 2>&1
gtime -v python3 ./examples/MNIST/mnist.py >> ./examples/MNIST/mnist_results.txt 2>&1