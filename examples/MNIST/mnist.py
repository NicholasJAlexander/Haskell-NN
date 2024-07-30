import os
import numpy as np
import struct
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning
import time

# Set the number of threads to 1
os.environ['OMP_NUM_THREADS'] = '1'

warnings.simplefilter("ignore", ConvergenceWarning)

def read_idx(filename):
    """Reads MNIST data from IDX file format."""
    with open(filename, 'rb') as file:
        zero, data_type, dims = struct.unpack('>HBB', file.read(4))
        shape = tuple(struct.unpack('>I', file.read(4))[0] for d in range(dims))
        return np.frombuffer(file.read(), dtype=np.uint8).reshape(shape)

def load_data():
    """Load training and test data from IDX files."""
    train_images = read_idx("./examples/MNIST/train-images-idx3-ubyte")
    train_labels = read_idx("./examples/MNIST/train-labels-idx1-ubyte")
    test_images = read_idx("./examples/MNIST/t10k-images-idx3-ubyte")
    test_labels = read_idx("./examples/MNIST/t10k-labels-idx1-ubyte")
    
    # Reshape images and normalize
    train_images = train_images.reshape((60000, 784)).astype('float32') / 255
    test_images = test_images.reshape((10000, 784)).astype('float32') / 255
    
    return train_images, train_labels, test_images, test_labels

def custom_accuracy_score(y_true, y_pred):
    """Calculate accuracy by comparing predicted labels with true labels."""
    correct_count = np.sum(y_pred == y_true)
    accuracy = correct_count / len(y_true)
    return accuracy

def train_network_epoch_by_epoch(X_train, y_train, X_test, y_test, max_epochs=100):
    """Train a neural network, one epoch at a time, printing accuracy each epoch."""
    mlp = MLPClassifier(hidden_layer_sizes=(28,), activation='tanh',
                        solver='sgd', random_state=1, learning_rate_init=0.1,
                        max_iter=1, warm_start=True)  # Setup for incremental learning

    for epoch in range(max_epochs):
        mlp.fit(X_train, y_train)  # Train one epoch
        y_pred = mlp.predict(X_test)
        accuracy = custom_accuracy_score(y_test, y_pred)
        print(f"Epoch {epoch + 1} - Test Accuracy: {accuracy * 100:.2f}%")
    
    return mlp

def main():
    train_images, train_labels, test_images, test_labels = load_data()
    
    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_images)
    X_test = scaler.transform(test_images)
    
    # Train the network and print accuracy each epoch
    start_time = time.time()
    train_network_epoch_by_epoch(X_train, train_labels, X_test, test_labels, max_epochs=10)
    print(f"Total training time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
