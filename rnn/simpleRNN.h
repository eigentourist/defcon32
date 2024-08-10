#ifndef RNN_H
#define RNN_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>

// Activation Function and its Derivative
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float sigmoid_derivative(float x) {
    return x * (1.0f - x);
}

inline float tanh_activation(float x) {
    return std::tanh(x);
}

inline float tanh_derivative(float x) {
    return 1.0f - x * x;
}

inline float relu_activation(float x) {
    return std::max(0.0f, x);
}

inline float relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

class TempBatch {
public:
    int id;
    std::vector<float> temps;
    std::vector<int> is_spring;
    std::vector<int> is_summer;
    std::vector<int> is_fall;
    std::vector<int> is_winter;
    std::vector<int> year;
    std::vector<float>avg3;
    std::vector<float>avg7;
    float target;
};

struct FeatureBatch {
    std::vector<std::vector<float>> features;  // Contains all the feature vectors (e.g., temps, is_spring, etc.)
    float target;  // The target value (e.g., next temperature reading)
};


class SimpleRNN {
public:
    SimpleRNN(int input_size, int hidden_size, int output_size);

    void forward(const std::vector<std::vector<float>>& input_sequence);
    void backward(const std::vector<float>& target, float learning_rate, const std::vector<std::vector<float>>& input_sequence);
    FeatureBatch convertTempBatchToFeatureBatch(const TempBatch& tempBatch);
    void train(std::vector<TempBatch>& data, int epochs, float learning_rate);
    float predict(const std::vector<std::vector<float>>& input_sequence);
    static void read_csv(const std::string& filename,
                               std::vector<float>& temps,
                               std::vector<float>& is_spring,
                               std::vector<float>& is_summer,
                               std::vector<float>& is_fall,
                               std::vector<float>& is_winter,
                               std::vector<float>& year,
                               std::vector<float>& avg3,
                               std::vector<float>& avg7);

private:
    int input_size, hidden_size, output_size;
    std::vector<float> hidden_state;
    std::vector<float> output;

    std::vector<std::vector<float>> Wxh, Whh, Why;
    std::vector<float> bh, by;

    void initialize_weights();
    void reset_hidden_state();
};

#endif // RNN_H
