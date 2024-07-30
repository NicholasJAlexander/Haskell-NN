import warnings
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
import time

# Ignore ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)

def load_and_preprocess_data():
    # Load Iris dataset
    data = load_iris()
    X, y = data.data, data.target

    # Normalize features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # Encode labels to one-hot
    encoder = OneHotEncoder()
    y_one_hot = encoder.fit_transform(y.reshape(-1, 1)).toarray()  # Ensure output is a dense numpy array

    return X_normalized, y_one_hot, y

def train_test_split_stratified(X, y_one_hot, y, test_size=0.2):
    # Stratified train-test split using original y for stratification
    X_train, X_test, y_train_one_hot, y_test_one_hot = train_test_split(
        X, y_one_hot, test_size=test_size, random_state=42, stratify=y)
    return X_train, X_test, y_train_one_hot, y_test_one_hot

def train_network_epoch_by_epoch(X_train, y_train, X_test, y_test, hidden_layer_sizes=(10,), learning_rate_init=0.05, max_iter=200):
    # Define all possible classes (for Iris dataset: 0, 1, 2)
    classes = np.array([0, 1, 2])
    
    # Create a neural network
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                        activation='logistic', solver='adam',
                        random_state=1, learning_rate_init=learning_rate_init,
                        max_iter=1, warm_start=True)  # Use max_iter=1 for one epoch at a time, warm_start=True to keep training between calls

    # Manually iterate through each epoch
    for epoch in range(max_iter):
        mlp.partial_fit(X_train, y_train, classes=classes)  # Provide all classes explicitly
        y_pred = mlp.predict(X_test)  # Predict on test data
        accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))  # Calculate accuracy
        print(f"Epoch {epoch + 1}/{max_iter} - Test Accuracy: {accuracy * 100:.2f}%")

    return mlp



def main():
    X, y_one_hot, y = load_and_preprocess_data()
    X_train, X_test, y_train_one_hot, y_test_one_hot = train_test_split_stratified(X, y_one_hot, y, test_size=0.2)
    
    # Train the network epoch by epoch and print test accuracy each epoch
    x = time.time()
    trained_net = train_network_epoch_by_epoch(X_train, y_train_one_hot, X_test, y_test_one_hot,
                                               hidden_layer_sizes=(10,), learning_rate_init=0.05, max_iter=100)
    print(f"time {time.time() - x}")
if __name__ == "__main__":
    main()
