#include "simpleRNN.h"
#include <algorithm>  // For std::shuffle
#include <random>     // For std::random_device and std::mt19937

SimpleRNN::SimpleRNN(int input_size, int hidden_size, int output_size)
    : input_size(input_size), hidden_size(hidden_size), output_size(output_size),
      hidden_state(hidden_size, 0.0f), output(output_size, 0.0f) {
    initialize_weights();
}

void SimpleRNN::initialize_weights() {
    std::random_device rd;
    std::mt19937 gen(rd());

    // Resize matrices
    Wxh.resize(hidden_size, std::vector<float>(input_size));
    Whh.resize(hidden_size, std::vector<float>(hidden_size));
    Why.resize(output_size, std::vector<float>(hidden_size));
    bh.resize(hidden_size);
    by.resize(output_size);

    // He initialization for Wxh (weights from input to hidden layer)
    float stddev_Wxh = std::sqrt(2.0f / input_size);
    std::normal_distribution<> dis_Wxh(0, stddev_Wxh);

    for (int i = 0; i < hidden_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            Wxh[i][j] = dis_Wxh(gen);
        }
    }

    // He initialization for Whh (weights from hidden to hidden layer)
    float stddev_Whh = std::sqrt(2.0f / hidden_size);
    std::normal_distribution<> dis_Whh(0, stddev_Whh);

    for (int i = 0; i < hidden_size; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            Whh[i][j] = dis_Whh(gen);
        }
    }

    // He initialization for Why (weights from hidden to output layer)
    float stddev_Why = std::sqrt(2.0f / hidden_size);
    std::normal_distribution<> dis_Why(0, stddev_Why);

    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            Why[i][j] = dis_Why(gen);
        }
    }

    // Initialize biases to zero
    bh.assign(hidden_size, 0.0f);
    by.assign(output_size, 0.0f);
}


void SimpleRNN::forward(const std::vector<std::vector<float>>& input_sequence) {
    std::vector<float> new_hidden_state(hidden_size, 0.0f);
    hidden_state.assign(hidden_size, 0.0f);  // Reset hidden state at the start of each sequence
    hidden_states.clear();
    hidden_states.push_back(hidden_state);

    // Iterate over each time step in the input sequence
    for (const auto& input_features : input_sequence) {
        for (int i = 0; i < hidden_size; ++i) {
            float activation = 0.0f;
            // Input-to-hidden connections
            for (int j = 0; j < input_size; ++j) {
                activation += Wxh[i][j] * input_features[j];
            }
            // Hidden-to-hidden connections
            for (int j = 0; j < hidden_size; ++j) {
                activation += Whh[i][j] * hidden_state[j];
            }
            activation += bh[i];
            new_hidden_state[i] = relu_activation(activation);
        }

        // Update hidden state for the next time step
        hidden_state = new_hidden_state;
        hidden_states.push_back(hidden_state);
    }

    // Compute the output layer's activations
    for (int i = 0; i < output_size; ++i) {
        float activation = 0.0f;
        for (int j = 0; j < hidden_size; ++j) {
            activation += Why[i][j] * hidden_state[j];
        }
        activation += by[i];
        output[i] = activation;  // No activation function for the output layer
    }
}



void SimpleRNN::backward(const std::vector<float>& target, float learning_rate, const std::vector<std::vector<float>>& input_sequence) {
    std::vector<float> output_error(output_size, 0.0f);
    std::vector<std::vector<float>> dWhy(output_size, std::vector<float>(hidden_size, 0.0f));
    std::vector<float> dby(output_size, 0.0f);
    std::vector<std::vector<float>> dWxh(hidden_size, std::vector<float>(input_size, 0.0f));
    std::vector<std::vector<float>> dWhh(hidden_size, std::vector<float>(hidden_size, 0.0f));
    std::vector<float> dbh(hidden_size, 0.0f);
    std::vector<float> hidden_gradient(hidden_size, 0.0f);

    const float clip_value = 0.5f;  // Adjust this value if necessary

    // Output layer error (no activation function, so no derivative needed)
    for (int i = 0; i < output_size; ++i) {
        output_error[i] = target[i] - output[i];
        dby[i] = output_error[i];
        for (int j = 0; j < hidden_size; ++j) {
            dWhy[i][j] = output_error[i] * hidden_states.back()[j];
            hidden_gradient[j] += output_error[i] * Why[i][j];
        }
    }

    // Backpropagate the final-output error through every time step.
    for (int t = static_cast<int>(input_sequence.size()) - 1; t >= 0; --t) {
        std::vector<float> activation_gradient(hidden_size, 0.0f);
        for (int i = 0; i < hidden_size; ++i) {
            activation_gradient[i] = hidden_gradient[i] * relu_derivative(hidden_states[t + 1][i]);
            dbh[i] += activation_gradient[i];
            for (int j = 0; j < input_size; ++j) {
                dWxh[i][j] += activation_gradient[i] * input_sequence[t][j];
            }
            for (int j = 0; j < hidden_size; ++j) {
                dWhh[i][j] += activation_gradient[i] * hidden_states[t][j];
            }
        }

        std::vector<float> previous_hidden_gradient(hidden_size, 0.0f);
        for (int j = 0; j < hidden_size; ++j) {
            for (int i = 0; i < hidden_size; ++i) {
                previous_hidden_gradient[j] += Whh[i][j] * activation_gradient[i];
            }
        }
        hidden_gradient = previous_hidden_gradient;
    }

    auto clipped_update = [learning_rate, clip_value](float gradient) {
        return std::max(-clip_value, std::min(clip_value, learning_rate * gradient));
    };

    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            Why[i][j] += clipped_update(dWhy[i][j]);
        }
        by[i] += clipped_update(dby[i]);
    }
    for (int i = 0; i < hidden_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            Wxh[i][j] += clipped_update(dWxh[i][j]);
        }
        for (int j = 0; j < hidden_size; ++j) {
            Whh[i][j] += clipped_update(dWhh[i][j]);
        }
        bh[i] += clipped_update(dbh[i]);
    }
}


FeatureBatch SimpleRNN::convertTempBatchToFeatureBatch(const TempBatch& tempBatch) {
    FeatureBatch featureBatch;
    
    // Combine all the vectors into the features vector
    size_t sequence_length = tempBatch.temps.size();  // Assuming all vectors have the same length
    
    for (size_t i = 0; i < sequence_length; ++i) {
        std::vector<float> feature_vector;
        
        feature_vector.push_back(tempBatch.temps[i]);
        feature_vector.push_back(static_cast<float>(tempBatch.is_spring[i]));
        feature_vector.push_back(static_cast<float>(tempBatch.is_summer[i]));
        feature_vector.push_back(static_cast<float>(tempBatch.is_fall[i]));
        feature_vector.push_back(static_cast<float>(tempBatch.is_winter[i]));
        feature_vector.push_back(static_cast<float>(tempBatch.year[i]));
        feature_vector.push_back(tempBatch.avg3[i]);
        feature_vector.push_back(tempBatch.avg7[i]);
        
        featureBatch.features.push_back(feature_vector);
    }
    
    featureBatch.target = tempBatch.target;
    
    return featureBatch;
}


void SimpleRNN::train(std::vector<TempBatch>& data, int epochs, float learning_rate) {
    constexpr size_t minimum_batches = 100;
    if (data.size() < minimum_batches) {
        std::cerr << "Error: a minimum of " << minimum_batches
                  << " temperature batches is necessary for this demonstration; found "
                  << data.size() << "." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // std::cout << "------- Epoch " << epoch << " -------" << std::endl;

        std::random_device rd;
        std::mt19937 g(rd());

        // Shuffle the batch_list
        std::shuffle(data.begin(), data.end(), g);

        // Now, select the first 50 batches from the shuffled list
        std::vector<TempBatch> selected_batches(data.begin(), data.begin() + 100);

        std::cout << selected_batches.size() << " selected batches." << std::endl;

        // Convert selected TempBatch objects to FeatureBatch objects
        std::vector<FeatureBatch> feature_batches;
        for (const auto& tempBatch : selected_batches) {
            feature_batches.push_back(convertTempBatchToFeatureBatch(tempBatch));
        }

        float loss = 0.0f;
        for (size_t i = 0; i < feature_batches.size(); ++i) {
            // Pass the feature vectors to the forward method
            forward(feature_batches[i].features);
            backward({feature_batches[i].target}, learning_rate, feature_batches[i].features);

            // Output size is 1 here, because we are predicting the next temp
            loss += 0.5f * std::pow(data[i].target - output[0], 2);
            // std::cout << "(Epoch " << epoch << ") temp " <<  data[i].target << ", predicted " << output[0] << std::endl;
        }
        std::cout << std::endl;
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << ", Loss: " << loss << std::endl;
    }
}


// float SimpleRNN::predict(const std::vector<std::vector<float>>& input_sequence) {
//     forward(input_sequence);
//     return output[0]; // Assuming output_size is 1 for this regression problem
// }


void SimpleRNN::read_csv(const std::string& filename,
                               std::vector<float>& temps,
                               std::vector<float>& is_spring,
                               std::vector<float>& is_summer,
                               std::vector<float>& is_fall,
                               std::vector<float>& is_winter,
                               std::vector<float>& year,
                               std::vector<float>& avg3,
                               std::vector<float>& avg7) {
    std::ifstream file(filename);
    std::string line;
    std::string is_fall_str;
    std::string is_spring_str;
    std::string is_summer_str;
    std::string is_winter_str;
    std::string year_str;
    std::string temp_str;
    std::string avg3_str;
    std::string avg7_str;

    if (!file.is_open()) {
        std::cerr << "Error: unable to open " << filename << "." << std::endl;
        exit(-1);
    }

    // Skip header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::getline(ss, is_fall_str, ',');
        std::getline(ss, is_spring_str, ',');
        std::getline(ss, is_summer_str, ',');
        std::getline(ss, is_winter_str, ',');
        std::getline(ss, year_str, ',');
        std::getline(ss, temp_str, ',');
        std::getline(ss, avg3_str, ',');
        std::getline(ss, avg7_str, ',');

        is_fall.push_back({ std::stof(is_fall_str) });
        is_spring.push_back({ std::stof(is_spring_str) });
        is_summer.push_back({ std::stof(is_summer_str) });
        is_winter.push_back({ std::stof(is_winter_str) });
        year.push_back({ std::stof(year_str) });
        temps.push_back({ std::stof(temp_str) });
        avg3.push_back({ std::stof(avg3_str) });
        avg7.push_back({ std::stof(avg7_str) });
    }

    std::cout << "Loaded " << is_spring.size() << " is_spring flags." << std::endl;
    std::cout << "Loaded " << is_summer.size() << " is_summer flags." << std::endl;
    std::cout << "Loaded " << is_fall.size() << " is_fall flags." << std::endl;
    std::cout << "Loaded " << is_winter.size() << " is_winter flags." << std::endl;
    std::cout << "Loaded " << temps.size() << " temp readings." << std::endl;
    std::cout << "Loaded " << avg3.size() << " avg3 values." << std::endl;
    std::cout << "Loaded " << avg7.size() << " avg7 values." << std::endl;
}
