import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * 0.1
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * 0.1
        self.bias_hidden = np.random.randn(1, self.hidden_size) * 0.1
        self.bias_output = np.random.randn(1, self.output_size) * 0.1

    # Feedforward pass
    def feedforward(self, X):
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)
        
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_layer_input)
        return self.output

    # Backpropagation to adjust weights
    def backpropagate(self, X, y, learning_rate=0.01):
        # Calculate output layer error
        output_error = y - self.output
        output_delta = output_error * sigmoid_derivative(self.output)

        # Calculate hidden layer error
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_layer_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_layer_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    # Training the network
    def train(self, X_train, y_train, epochs=1000, learning_rate=0.01):
        loss_history = []  # To track loss over epochs
        for epoch in range(epochs):
            self.feedforward(X_train)
            self.backpropagate(X_train, y_train, learning_rate)
            
            # Calculate loss (mean squared error)
            loss = np.mean(np.square(y_train - self.output))
            loss_history.append(loss)

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')
        
        # Plot loss history to see the learning progress
        plt.plot(range(epochs), loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.show()

    # Predicting class labels
    def predict(self, X):
        predictions = self.feedforward(X)
        return np.argmax(predictions, axis=1)

# Function to load and preprocess dataset
def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Split features and labels
    X = df.iloc[:, :-1].values  # All rows, all columns except the last one
    y = df.iloc[:, -1].values   # All rows, only the last column (target labels)

    # Encode the labels (Iris-setosa, Iris-versicolor, Iris-virginica)
    y = pd.get_dummies(y).values

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Main execution
if __name__ == "__main__":
    # Input the path of the dataset
    file_path = "/home/cs-ai-07/merin/exp14/Iris.csv"

    # Load and preprocess the data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

    # Define the size of the neural network
    input_size = X_train.shape[1]  # Number of features
    hidden_size = 5  # Number of neurons in the hidden layer
    output_size = y_train.shape[1]  # Number of classes

    # Create and train the neural network
    nn = NeuralNetwork(input_size, hidden_size, output_size)
    nn.train(X_train, y_train, epochs=1000, learning_rate=0.01)

    # Test the network
    y_pred = nn.predict(X_test)
    y_test_classes = np.argmax(y_test, axis=1)  # Convert one-hot encoding to class labels

    accuracy = accuracy_score(y_test_classes, y_pred)
    print(f"Accuracy on test set: {accuracy * 100:.2f}%")
