import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Force PyTorch to use a single thread
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Force PyTorch to use a single thread
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'


def create_network(neurons, activation_func):
    layers = []
    for i in range(len(neurons) - 1):
        layers.append(nn.Linear(neurons[i], neurons[i+1]))
        layers.append(activation_func())
    return nn.Sequential(*layers)

def mse(output, target):
    return ((output - target) ** 2).mean()

def main():
    args = sys.argv[1:]  # Get command line arguments
    print(args)  # Print the raw arguments to see what is received
    if len(args) == 3:
        neurons = list(map(int, args[0].split(',')))
        batch_size = int(args[1])
        epochs = int(args[2])

        # Create the neural network
        model = create_network(neurons, nn.Tanh)
        optimizer = optim.SGD(model.parameters(), lr=0.0000001)
        loss_func = nn.MSELoss()

        # Prepare dummy data (as in the Haskell example, using 1s)
        input = torch.rand(batch_size, neurons[0])

        # For target tensor
        target = torch.rand(batch_size, neurons[-1])

        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(input)
            loss = loss_func(outputs, target)
            loss.backward()
            optimizer.step()
            
            #print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


    else:
        print("Usage: program <neurons> <batchSize> <epochs>")

if __name__ == "__main__":
    main()
