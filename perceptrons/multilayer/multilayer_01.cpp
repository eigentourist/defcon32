#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

// Activation function (sigmoid)
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// Derivative of sigmoid function
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

class Neuron {
public:
    std::vector<double> weights;
    double bias;
    double output;

    Neuron(int inputs) : bias(0) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        for (int i = 0; i < inputs; ++i) {
            weights.push_back(dis(gen));
        }
    }
};

class Layer {
public:
    std::vector<Neuron> neurons;

    Layer(int num_neurons, int inputs_per_neuron) {
        for (int i = 0; i < num_neurons; ++i) {
            neurons.emplace_back(inputs_per_neuron);
        }
    }
};

class MLP {
public:
    std::vector<Layer> layers;

    MLP(const std::vector<int>& layer_sizes) {
        for (size_t i = 1; i < layer_sizes.size(); ++i) {
            layers.emplace_back(layer_sizes[i], layer_sizes[i-1]);
        }
    }

    std::vector<double> forward(const std::vector<double>& inputs) {
        std::vector<double> current_inputs = inputs;
        for (auto& layer : layers) {
            std::vector<double> layer_outputs;
            for (auto& neuron : layer.neurons) {
                double sum = 0.0;
                for (size_t i = 0; i < current_inputs.size(); ++i) {
                    sum += current_inputs[i] * neuron.weights[i];
                }
                sum += neuron.bias;
                neuron.output = sigmoid(sum);
                layer_outputs.push_back(neuron.output);
            }
            current_inputs = layer_outputs;
        }
        return current_inputs;
    }

    void train(const std::vector<std::vector<double>>& training_inputs,
               const std::vector<double>& training_outputs,
               int epochs, double learning_rate) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_error = 0.0;
            for (size_t i = 0; i < training_inputs.size(); ++i) {
                auto output = forward(training_inputs[i]);
                std::vector<std::vector<double>> deltas(layers.size());

                // Calculate output layer deltas
                double error = training_outputs[i] - output[0];
                total_error += std::abs(error);
                deltas.back().push_back(error * sigmoid_derivative(output[0]));

                // Calculate hidden layer deltas
                for (int l = layers.size() - 2; l >= 0; --l) {
                    for (size_t j = 0; j < layers[l].neurons.size(); ++j) {
                        double error = 0.0;
                        for (size_t k = 0; k < layers[l+1].neurons.size(); ++k) {
                            error += deltas[l+1][k] * layers[l+1].neurons[k].weights[j];
                        }
                        deltas[l].push_back(error * sigmoid_derivative(layers[l].neurons[j].output));
                    }
                }

                // Update weights and biases
                for (size_t l = 0; l < layers.size(); ++l) {
                    std::vector<double> inputs = (l == 0) ? training_inputs[i] : std::vector<double>(layers[l-1].neurons.size());
                    if (l > 0) {
                        for (size_t j = 0; j < layers[l-1].neurons.size(); ++j) {
                            inputs[j] = layers[l-1].neurons[j].output;
                        }
                    }
                    for (size_t j = 0; j < layers[l].neurons.size(); ++j) {
                        for (size_t k = 0; k < inputs.size(); ++k) {
                            layers[l].neurons[j].weights[k] += learning_rate * deltas[l][j] * inputs[k];
                        }
                        layers[l].neurons[j].bias += learning_rate * deltas[l][j];
                    }
                }
            }
            if (epoch % 100 == 0) {
                std::cout << "Epoch " << epoch << ", Error: " << total_error << std::endl;
            }
        }
    }
};


std::vector<std::vector<double>> read_csv(const std::string& filename) {
    std::vector<std::vector<double>> data;
    std::ifstream file(filename);
    std::string line;

    // Skip the header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ',')) {
            try {
                // Trim whitespace
                cell.erase(0, cell.find_first_not_of(" \t\n\r\f\v"));
                cell.erase(cell.find_last_not_of(" \t\n\r\f\v") + 1);

                // Convert 'True' to 1 and 'False' to 0
                if (cell == "True") {
                    row.push_back(1.0);
                } else if (cell == "False") {
                    row.push_back(0.0);
                } else if (!cell.empty()) {
                    row.push_back(std::stod(cell));
                } else {
                    // Handle empty cells by pushing a default value (e.g., 0)
                    row.push_back(0.0);
                }
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid argument: " << cell << " in file " << filename << std::endl;
                // You might want to handle this error differently
                row.push_back(0.0);
            } catch (const std::out_of_range& e) {
                std::cerr << "Out of range: " << cell << " in file " << filename << std::endl;
                // You might want to handle this error differently
                row.push_back(0.0);
            }
        }
        data.push_back(row);
    }

    return data;
}


void evaluate_model(MLP& mlp, const std::vector<std::vector<double>>& inputs, const std::vector<double>& expected_outputs) {
    int correct = 0;
    int total = inputs.size();
    int true_positives = 0, false_positives = 0, true_negatives = 0, false_negatives = 0;

    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = mlp.forward(inputs[i]);
        int predicted = (output[0] > 0.5) ? 1 : 0;
        int expected = expected_outputs[i];

        if (predicted == expected) {
            correct++;
        }

        if (predicted == 1 && expected == 1) true_positives++;
        if (predicted == 1 && expected == 0) false_positives++;
        if (predicted == 0 && expected == 0) true_negatives++;
        if (predicted == 0 && expected == 1) false_negatives++;
    }

    double accuracy = static_cast<double>(correct) / total;

    // Metrics for class 1 (Flare)
    double precision_flare = true_positives / static_cast<double>(true_positives + false_positives);
    double recall_flare = true_positives / static_cast<double>(true_positives + false_negatives);
    double f1_score_flare = 2 * (precision_flare * recall_flare) / (precision_flare + recall_flare);

    // Metrics for class 0 (No Flare)
    double precision_no_flare = true_negatives / static_cast<double>(true_negatives + false_negatives);
    double recall_no_flare = true_negatives / static_cast<double>(true_negatives + false_positives);
    double f1_score_no_flare = 2 * (precision_no_flare * recall_no_flare) / (precision_no_flare + recall_no_flare);

    std::cout << "Accuracy: " << accuracy << std::endl;
    std::cout << "Metrics for Flare (Class 1):" << std::endl;
    std::cout << "Precision: " << precision_flare << std::endl;
    std::cout << "Recall: " << recall_flare << std::endl;
    std::cout << "F1 Score: " << f1_score_flare << std::endl;

    std::cout << "\nMetrics for No Flare (Class 0):" << std::endl;
    std::cout << "Precision: " << precision_no_flare << std::endl;
    std::cout << "Recall: " << recall_no_flare << std::endl;
    std::cout << "F1 Score: " << f1_score_no_flare << std::endl;

    std::cout << "\nConfusion Matrix:" << std::endl;
    std::cout << "True Positives: " << true_positives << " | False Negatives: " << false_negatives << std::endl;
    std::cout << "False Positives: " << false_positives << " | True Negatives: " << true_negatives << std::endl;
}


int main() {
    // Read training data
    auto training_data = read_csv("data/flare_train_balanced.csv");
    // auto training_data = read_csv("data/flare_train_top_features.csv");
    std::vector<std::vector<double>> training_inputs;
    std::vector<double> training_outputs;
    for (const auto& row : training_data) {
        training_inputs.push_back(std::vector<double>(row.begin(), row.end() - 1));
        training_outputs.push_back(row.back());
    }

    // Read test data
    auto test_data = read_csv("data/flare_test.csv");
    // auto test_data = read_csv("data/flare_test_top_features.csv");
    std::vector<std::vector<double>> test_inputs;
    std::vector<double> test_outputs;
    for (const auto& row : test_data) {
        test_inputs.push_back(std::vector<double>(row.begin(), row.end() - 1));
        test_outputs.push_back(row.back());
    }

    // Create and train the MLP
    MLP solar_flare_mlp({30, 20, 10, 1});  // 30 input features, two hidden layers (20 and 10 neurons), 1 output neuron
    // MLP solar_flare_mlp({4, 12, 8, 1});  // 4 input features, two hidden layers (12 and 8 neurons), 1 output neuron
    solar_flare_mlp.train(training_inputs, training_outputs, 2000, 0.01);  // Increased epochs, reduced learning rate

    // Evaluate the model
    std::cout << "\nTraining set performance:" << std::endl;
    evaluate_model(solar_flare_mlp, training_inputs, training_outputs);
    std::cout << "\nTest set performance:" << std::endl;
    evaluate_model(solar_flare_mlp, test_inputs, test_outputs);

    return 0;
}
