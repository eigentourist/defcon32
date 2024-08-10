#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <random>


// Activation function (Sigmoid) and its derivative
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// Class representing a single-layer neural network
class NeuralNetwork {
public:
    NeuralNetwork(int inputSize, int hiddenSize, int outputSize);

    void train(std::vector<std::vector<double>> &inputs, std::vector<std::vector<double>> &outputs, int epochs, double learningRate);
    std::vector<double> predict(std::vector<double> &input);

private:
    std::vector<std::vector<double>> weightsInputHidden;
    std::vector<std::vector<double>> weightsHiddenOutput;

    std::vector<double> hiddenLayer;
    std::vector<double> outputLayer;

    double randomWeight();
    void forward(std::vector<double> &input);
    void backward(std::vector<double> &input, std::vector<double> &output, double learningRate);
};

// Constructor initializes weights with random values
NeuralNetwork::NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
    weightsInputHidden.resize(inputSize, std::vector<double>(hiddenSize));
    weightsHiddenOutput.resize(hiddenSize, std::vector<double>(outputSize));
    hiddenLayer.resize(hiddenSize);
    outputLayer.resize(outputSize);

    for (int i = 0; i < inputSize; ++i)
        for (int j = 0; j < hiddenSize; ++j)
            weightsInputHidden[i][j] = randomWeight();

    for (int i = 0; i < hiddenSize; ++i)
        for (int j = 0; j < outputSize; ++j)
            weightsHiddenOutput[i][j] = randomWeight();
}

// Random weight initializer
double NeuralNetwork::randomWeight() {
    static std::mt19937 generator;
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    return distribution(generator);
}

// Forward propagation
void NeuralNetwork::forward(std::vector<double> &input) {
    // Calculate hidden layer activations
    for (int j = 0; j < hiddenLayer.size(); ++j) {
        hiddenLayer[j] = 0.0;
        for (int i = 0; i < input.size(); ++i) {
            hiddenLayer[j] += input[i] * weightsInputHidden[i][j];
        }
        hiddenLayer[j] = sigmoid(hiddenLayer[j]);
    }

    // Calculate output layer activations
    for (int j = 0; j < outputLayer.size(); ++j) {
        outputLayer[j] = 0.0;
        for (int i = 0; i < hiddenLayer.size(); ++i) {
            outputLayer[j] += hiddenLayer[i] * weightsHiddenOutput[i][j];
        }
        outputLayer[j] = sigmoid(outputLayer[j]);
    }
}

// Backward propagation
void NeuralNetwork::backward(std::vector<double> &input, std::vector<double> &output, double learningRate) {
    // Calculate output layer error and deltas
    std::vector<double> outputErrors(outputLayer.size());
    std::vector<double> outputDeltas(outputLayer.size());
    for (int i = 0; i < outputLayer.size(); ++i) {
        outputErrors[i] = output[i] - outputLayer[i];
        outputDeltas[i] = outputErrors[i] * sigmoid_derivative(outputLayer[i]);
    }

    // Calculate hidden layer error and deltas
    std::vector<double> hiddenErrors(hiddenLayer.size());
    std::vector<double> hiddenDeltas(hiddenLayer.size());
    for (int i = 0; i < hiddenLayer.size(); ++i) {
        hiddenErrors[i] = 0.0;
        for (int j = 0; j < outputLayer.size(); ++j) {
            hiddenErrors[i] += outputDeltas[j] * weightsHiddenOutput[i][j];
        }
        hiddenDeltas[i] = hiddenErrors[i] * sigmoid_derivative(hiddenLayer[i]);
    }

    // Update weights between hidden and output layers
    for (int i = 0; i < hiddenLayer.size(); ++i) {
        for (int j = 0; j < outputLayer.size(); ++j) {
            weightsHiddenOutput[i][j] += learningRate * outputDeltas[j] * hiddenLayer[i];
        }
    }

    // Update weights between input and hidden layers
    for (int i = 0; i < input.size(); ++i) {
        for (int j = 0; j < hiddenLayer.size(); ++j) {
            weightsInputHidden[i][j] += learningRate * hiddenDeltas[j] * input[i];
        }
    }
}

// Training function
void NeuralNetwork::train(std::vector<std::vector<double>> &inputs, std::vector<std::vector<double>> &outputs, int epochs, double learningRate) {
    double minError = std::numeric_limits<double>::max();
    int patience = 5;  // Number of epochs to wait for an improvement before stopping
    int wait = 0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalError = 0.0;

        for (int i = 0; i < inputs.size(); ++i) {
            forward(inputs[i]);
            backward(inputs[i], outputs[i], learningRate);

            // Calculate total error for monitoring
            for (int j = 0; j < outputs[i].size(); ++j) {
                totalError += std::pow(outputs[i][j] - outputLayer[j], 2);
            }
        }

        double avgError = totalError / inputs.size();
        std::cout << "Epoch " << epoch + 1 << " - Error: " << avgError << std::endl;

        // Early stopping check
        if (avgError < minError) {
            minError = avgError;
            wait = 0;  // Reset wait
        } else {
            wait++;
            if (wait >= patience) {
                std::cout << "Early stopping triggered at epoch " << epoch + 1 << std::endl;
                break;
            }
        }
    }
}

// Prediction function
std::vector<double> NeuralNetwork::predict(std::vector<double> &input) {
    forward(input);
    return outputLayer;
}

// Helper function to load data from CSV
std::vector<std::vector<double>> loadCSV(const std::string &filename, bool hasLabels = false) {
    std::ifstream file(filename);
    std::vector<std::vector<double>> data;

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;  // Return an empty vector
    }

    std::string line;
    bool firstLine = true;

    while (std::getline(file, line)) {
        if (firstLine && hasLabels) {
            firstLine = false;
            continue;
        }

        std::vector<double> row;
        std::stringstream ss(line);
        std::string value;

        while (std::getline(ss, value, ',')) {
            try {
                row.push_back(std::stod(value));  // Convert string to double
            } catch (const std::invalid_argument& e) {
                std::cerr << "Conversion error: " << e.what() << " for value '" << value << "'" << std::endl;
                return data;  // Return an empty vector on error
            }
        }

        data.push_back(row);
    }

    file.close();
    return data;
}


int main() {
    // Load data
    std::vector<std::vector<double>> train_data = loadCSV("data/flare_train.csv", true);
    std::vector<std::vector<double>> test_data = loadCSV("data/flare_test.csv", true);

    // Separate features and labels for training data
    std::vector<std::vector<double>> X_train;
    std::vector<std::vector<double>> y_train;

    for (const auto& row : train_data) {
        X_train.push_back(std::vector<double>(row.begin(), row.end() - 1));  // All columns except the last
        y_train.push_back(std::vector<double>{row.back()});  // The last column
    }

    // Separate features and labels for test data
    std::vector<std::vector<double>> X_test;
    std::vector<std::vector<double>> y_test;

    for (const auto& row : test_data) {
        X_test.push_back(std::vector<double>(row.begin(), row.end() - 1));  // All columns except the last
        y_test.push_back(std::vector<double>{row.back()});  // The last column
    }

    // Initialize the neural network with the chosen learning rate
    NeuralNetwork nn(X_train[0].size(), 10, 1);

    // Adjust the learning rate here
    double learningRate = 0.05;  // Start with a lower learning rate
    nn.train(X_train, y_train, 1000, learningRate);
    
    // Initialize confusion matrix counters
    int TP = 0, TN = 0, FP = 0, FN = 0;

    // Experiment with different thresholds
    double threshold = 0.7;  // Adjust this threshold to balance precision and recall

    // Test the neural network and populate confusion matrix
    for (int i = 0; i < X_test.size(); ++i) {
        std::vector<double> prediction = nn.predict(X_test[i]);
        double predicted_class = (prediction[0] >= threshold) ? 1.0 : 0.0;  // Apply adjustable threshold
        double actual_class = y_test[i][0];

        if (predicted_class == 1.0 && actual_class == 1.0) {
            TP++;
        } else if (predicted_class == 0.0 && actual_class == 0.0) {
            TN++;
        } else if (predicted_class == 1.0 && actual_class == 0.0) {
            FP++;
        } else if (predicted_class == 0.0 && actual_class == 1.0) {
            FN++;
        }
    }

    // Calculate precision and recall
    double precision = TP / double(TP + FP);
    double recall = TP / double(TP + FN);

    // Output results
    std::cout << "Confusion Matrix:" << std::endl;
    std::cout << "True Positive: " << TP << "  False Positive: " << FP << std::endl;
    std::cout << "False Negative: " << FN << "  True Negative: " << TN << std::endl;
    std::cout << "Precision: " << precision << std::endl;
    std::cout << "Recall: " << recall << std::endl;

    return 0;
}

