#!/bin/bash

# Fixed parameters
b=1000  # batch size
epochs=10  # number of epochs

cabal build test
# Forcing pytorch to use only 1 thread
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# Parameters

hidden_layers=(1 9 19)
num_neurons=(1 10 20 30)

# Loop to vary the depth of the network
echo "--------------------------Haskell--------------------------" >> ./test_stats/depth_neurons.txt 2>&1
for hidden_layer in "${hidden_layers[@]}"; do
    # Create a neuron string that increases the network depth
    for neurons_per_layer in "${num_neurons[@]}"; do
        neurons=""
        for i in $(seq 0 $hidden_layer); do
            neurons+="$neurons_per_layer,"
        done
        neurons=${neurons%,}  # Remove the trailing comma

        echo "Haskell: Stats for neurons: $neurons, batchsize: $b and epochs: $epochs" >> ./test_stats/depth_neurons.txt 2>&1
        gtime -f "Max memory: %M KB\nCPU usage: %P\nTime: %U" cabal run test "$neurons" "$b" "$epochs" >> ./test_stats/depth_neurons.txt 2>&1
    done
done

echo "--------------------------Python--------------------------" >> ./test_stats/depth_neurons.txt 2>&1
for hidden_layer in "${hidden_layers[@]}"; do
    # Create a neuron string that increases the network depth
    for neurons_per_layer in "${num_neurons[@]}"; do
        neurons=""
        for i in $(seq 0 $hidden_layer); do
            neurons+="$neurons_per_layer,"
        done
        neurons=${neurons%,}  # Remove the trailing comma

        echo "Python: Stats for neurons: $neurons, batchsize: $b and epochs: $epochs" >> ./test_stats/depth_neurons.txt 2>&1
        gtime -f "Max memory: %M KB\nCPU usage: %P\nTime: %U" python3 test.py "$neurons" "$b" "$epochs" >> ./test_stats/depth_neurons.txt 2>&1
    done
done
