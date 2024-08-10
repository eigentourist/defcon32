#include "simpleRNN.h"


int main() {
    // Initialize RNN parameters
    int input_size = 1;
    int hidden_size = 16;
    int output_size = 1;
    int epochs = 1000;
    float learning_rate = 0.001f;
    int sequence_length = 7;  // Use sequences of 7 days

    SimpleRNN rnn(input_size, hidden_size, output_size);
    std::cout << "RNN created." << std::endl;

    // Load the data from the CSV file
    std::vector<float> temps;
    std::vector<float> is_spring;
    std::vector<float> is_summer;
    std::vector<float> is_fall;
    std::vector<float> is_winter;
    std::vector<float> year;
    std::vector<float> avg3;
    std::vector<float> avg7;
    rnn.read_csv("data/temps-1hot.csv", temps, is_spring, is_summer, is_fall, is_winter, year, avg3, avg7);
    std::cout << std::endl;

    std::vector<TempBatch> batches;
    int id_counter = 1;  // To give each TempReading a unique id

    for (int i = 0; i <= temps.size() - sequence_length - 1; ++i) {  
        TempBatch t;
        t.id = id_counter++;  // Assign a unique id to each TempReading

        for (int j = i; j < i + sequence_length; ++j) {
            t.is_spring.push_back(is_spring[j]);
            t.is_summer.push_back(is_summer[j]);
            t.is_fall.push_back(is_fall[j]);
            t.is_winter.push_back(is_spring[j]);
            t.year.push_back(year[j]);
            t.avg3.push_back(avg3[j]);
            t.avg7.push_back(avg3[j]);
            t.temps.push_back(temps[j]);
        }

        t.target = temps[i + sequence_length];
        batches.push_back(t);  // Add the populated TempReading to the batch
        std::cout << ".";
    }

    std::cout << std::endl;

    // Test print of loaded batches
    // for (int i = 0; i < batches.size(); ++i) {
    //     std::cout << "id: " << batches[i].id << std::endl;
    //     for (int j = 0; j < batches[i].temps.size(); ++j) {
    //         std::cout << batches[i].is_spring[j] << " ";
    //         std::cout << batches[i].is_fall[j] << " ";
    //         std::cout << batches[i].is_summer[j] << " ";
    //         std::cout << batches[i].is_winter[j] << " ";
    //         std::cout << batches[i].year[j] << " ";
    //         std::cout << batches[i].temps[j] << " ";
    //         std::cout << batches[i].avg3[j] << " ";
    //         std::cout << batches[i].avg7[j] << " ";
    //     }
    //     std::cout << "  --- " << batches[i].target << std::endl;
    // }

    // Train the RNN using FeatureBatch objects
    rnn.train(batches, epochs, learning_rate);

    // Test the RNN on a sample input sequence
    // std::vector<std::vector<float>> test_input = {
    //     { 15.0f }, { 14.0f }, { 13.0f }, { 12.0f }, { 11.0f }, { 10.0f }, { 9.0f }
    // };
    // float prediction = rnn.predict(test_input);
    // std::cout << "Prediction for input sequence: " << prediction << std::endl;

    return 0;
}
