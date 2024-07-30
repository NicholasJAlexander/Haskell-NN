#!/bin/bash

# Build the iris project
cabal build iris

# Run the iris Haskell script
echo "Running iris.hs" >> ./examples/IRIS/iris_results.txt 2>&1
gtime -v cabal run iris >> ./examples/IRIS/iris_results.txt 2>&1

# Run the iris Python script
echo "Running iris.py" >> ./examples/IRIS/iris_results.txt 2>&1
gtime -v python3 ./examples/IRIS/iris.py >> ./examples/IRIS/iris_results.txt 2>&1
