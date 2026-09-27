import numpy as np

def relu(x):
    """ReLU activation function: f(x) = max(0, x)"""
    return np.maximum(0, x)

def neural_network(inputs, weights, biases):
    """
    Simple feedforward neural network.
    
    Architecture:
    - Input layer: 2 neurons
    - Hidden layer 1: 2 neurons (ReLU)
    - Hidden layer 2: 2 neurons (ReLU)
    - Output layer: 1 neuron (ReLU)
    
    Parameters:
    -----------
    inputs : np.ndarray
        Input data of shape (n_samples, 2)
    weights : list of np.ndarray
        Weight matrices for each layer
    biases : list of np.ndarray
        Bias vectors for each layer
    
    Returns:
    --------
    output : np.ndarray
        Network predictions
    """
    # Layer 1: input -> hidden 1
    layer1 = relu(np.dot(inputs, weights[0]) + biases[0])
    
    # Layer 2: hidden 1 -> hidden 2
    layer2 = relu(np.dot(layer1, weights[1]) + biases[1])
    
    # Output layer: hidden 2 -> output
    output = relu(np.dot(layer2, weights[2]) + biases[2])
    
    return output

# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Initialize weights and biases
    weights = [
        np.random.rand(2, 2),   # input -> hidden 1
        np.random.rand(2, 2),   # hidden 1 -> hidden 2
        np.random.rand(2, 1),   # hidden 2 -> output
    ]
    biases = [
        np.random.rand(2),      # hidden 1
        np.random.rand(2),      # hidden 2
        np.random.rand(1),      # output
    ]
    
    # XOR input data
    inputs = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    
    # Expected outputs (for reference, not used in forward pass)
    targets = np.array([[0], [1], [1], [0]])
    
    # Forward pass
    predictions = neural_network(inputs, weights, biases)
    
    print("Inputs:")
    print(inputs)
    print("\nTargets (expected):")
    print(targets)
    print("\nPredictions (untrained network):")
    print(predictions)
