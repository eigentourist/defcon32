#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define NUM_INPUTS 2
#define LEARNING_RATE 0.1
#define NUM_EPOCHS 10000  // Increased number of epochs

// Activation function (Step function)
int activate(double sum) {
    return sum > 0 ? 1 : 0;
}

// Train the perceptron
void train(double inputs[NUM_INPUTS], double weights[NUM_INPUTS], double* bias, int target) {
    double sum = 0;
    for (int i = 0; i < NUM_INPUTS; i++) {
        sum += inputs[i] * weights[i];
    }
    sum += *bias;
    
    int prediction = activate(sum);
    int error = target - prediction;
    
    // Update weights and bias
    for (int i = 0; i < NUM_INPUTS; i++) {
        weights[i] += LEARNING_RATE * error * inputs[i];
    }
    *bias += LEARNING_RATE * error;
}

// Predict using the trained perceptron
int predict(double inputs[NUM_INPUTS], double weights[NUM_INPUTS], double bias) {
    double sum = 0;
    for (int i = 0; i < NUM_INPUTS; i++) {
        sum += inputs[i] * weights[i];
    }
    sum += bias;
    return activate(sum);
}

int main() {
    srand(time(NULL));
    
    // Initialize weights and bias randomly
    double weights[NUM_INPUTS];
    for (int i = 0; i < NUM_INPUTS; i++) {
        weights[i] = ((double)rand() / RAND_MAX) * 2 - 1; // Random value between -1 and 1
    }
    double bias = ((double)rand() / RAND_MAX) * 2 - 1;
    
    // Training data for XOR gate
    double training_inputs[4][NUM_INPUTS] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    int training_outputs[4] = {0, 1, 1, 0};  // XOR outputs
    
    // Training
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        for (int i = 0; i < 4; i++) {
            train(training_inputs[i], weights, &bias, training_outputs[i]);
        }
    }
    
    // Testing
    printf("Testing the trained perceptron for XOR:\n");
    int correct = 0;
    for (int i = 0; i < 4; i++) {
        int result = predict(training_inputs[i], weights, bias);
        printf("Input: %.0f %.0f, Expected: %d, Output: %d\n", 
               training_inputs[i][0], training_inputs[i][1], training_outputs[i], result);
        if (result == training_outputs[i]) {
            correct++;
        }
    }
    
    printf("\nAccuracy: %d%%\n", (correct * 100) / 4);
    
    return 0;
}