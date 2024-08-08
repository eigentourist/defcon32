#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

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
               const std::vector<std::vector<double>>& training_outputs,
               int epochs, double learning_rate) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_error = 0.0;
            for (size_t i = 0; i < training_inputs.size(); ++i) {
                auto output = forward(training_inputs[i]);
                std::vector<std::vector<double>> deltas(layers.size());

                // Calculate output layer deltas
                for (size_t j = 0; j < layers.back().neurons.size(); ++j) {
                    double error = training_outputs[i][j] - output[j];
                    total_error += std::abs(error);
                    deltas.back().push_back(error * sigmoid_derivative(output[j]));
                }

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

void evaluate_model(MLP& mlp, const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& expected_outputs) {
    std::cout << "Model Evaluation:" << std::endl;
    std::cout << "Inputs\t\tExpected\tOutput" << std::endl;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = mlp.forward(inputs[i]);
        std::cout << inputs[i][0] << ", " << inputs[i][1] << "\t\t" 
                  << expected_outputs[i][0] << "\t\t" 
                  << std::round(output[0] * 100.0) / 100.0 << std::endl;
    }
}

int main() {
    MLP xor_mlp({2, 2, 1});  // 2 input neurons, 2 hidden neurons, 1 output neuron

    std::vector<std::vector<double>> training_inputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    std::vector<std::vector<double>> training_outputs = {
        {0}, {1}, {1}, {0}
    };

    xor_mlp.train(training_inputs, training_outputs, 10000, 0.1);

    evaluate_model(xor_mlp, training_inputs, training_outputs);

    return 0;
}
