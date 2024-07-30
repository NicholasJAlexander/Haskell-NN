#!/bin/bash
cabal build test
# Forcing pytorch to use only 1 thread
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# Parameters
neurons="20,20,20"

batch_sizes=(1 5000 10000)
num_epochs=(1 10 20 30)

# Loop over batch sizes using seq for step increments
echo "--------------------------Haskell--------------------------" >> ./test_stats/epochs_batch.txt 2>&1
for b in "${batch_sizes[@]}"; do
    for epochs in "${num_epochs[@]}"; do
        echo "Haskell: Stats for neurons: $neurons, batchsize: $b and epochs: $epochs"  >> ./test_stats/epochs_batch.txt 2>&1
        gtime -f "Max memory: %M KB\nCPU usage: %P\nTime: %U" cabal run test "$neurons" "$b" "$epochs" >> ./test_stats/epochs_batch.txt 2>&1 
    done
done
#python


echo "--------------------------Python--------------------------" >> ./test_stats/epochs_batch.txt 2>&1
for b in "${batch_sizes[@]}"; do
    for epochs in "${num_epochs[@]}"; do
        echo "Python: Stats for neurons: $neurons, batchsize: $b and epochs: $epochs"  >> ./test_stats/epochs_batch.txt 2>&1
        gtime -f "Max memory: %M KB\nCPU usage: %P\nTime: %U" python3 test.py "$neurons" "$b" "$epochs" >> ./test_stats/epochs_batch.txt 2>&1
    done
done
