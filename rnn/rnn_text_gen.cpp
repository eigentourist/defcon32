#include <iostream>
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include <cmath>
#include <random>

// Function to apply softmax
Eigen::VectorXd softmax(const Eigen::VectorXd& x) {
    Eigen::VectorXd exp_x = x.array().exp();
    return exp_x / exp_x.sum();
}

// Function to calculate the loss and gradients
double lossFun(
    const std::vector<int>& inputs,
    const std::vector<int>& targets,
    Eigen::MatrixXd& Wxh,
    Eigen::MatrixXd& Whh,
    Eigen::MatrixXd& Why,
    Eigen::VectorXd& bh,
    Eigen::VectorXd& by,
    Eigen::VectorXd& hprev,
    Eigen::MatrixXd& dWxh,
    Eigen::MatrixXd& dWhh,
    Eigen::MatrixXd& dWhy,
    Eigen::VectorXd& dbh,
    Eigen::VectorXd& dby
) {
    std::vector<Eigen::VectorXd> xs(inputs.size());
    std::vector<Eigen::VectorXd> hs(inputs.size() + 1);
    std::vector<Eigen::VectorXd> ys(inputs.size());
    std::vector<Eigen::VectorXd> ps(inputs.size());
    hs[0] = hprev;
    double loss = 0.0;

    // Forward pass
    for (size_t t = 0; t < inputs.size(); ++t) {
        xs[t] = Eigen::VectorXd::Zero(Wxh.cols());
        xs[t](inputs[t]) = 1.0;
        hs[t + 1] = (Wxh * xs[t] + Whh * hs[t] + bh).array().tanh();
        ys[t] = Why * hs[t + 1] + by;
        ps[t] = softmax(ys[t]);
        loss += -std::log(ps[t](targets[t]));
    }

    // Backward pass: compute gradients
    Eigen::VectorXd dhnext = Eigen::VectorXd::Zero(Whh.rows());
    double max_gradient = 5.0; // Maximum allowable gradient value

    for (int t = inputs.size() - 1; t >= 0; --t) {
        Eigen::VectorXd dy = ps[t];
        dy(targets[t]) -= 1.0;

        dWhy += dy * hs[t + 1].transpose();
        dby += dy;

        Eigen::VectorXd dh = Why.transpose() * dy + dhnext;
        Eigen::VectorXd dhraw = (1.0 - hs[t + 1].array().square()).matrix().cwiseProduct(dh);
        dbh += dhraw;
        dWxh += dhraw * xs[t].transpose();
        dWhh += dhraw * hs[t].transpose();
        dhnext = Whh.transpose() * dhraw;

        // Apply gradient clipping
        dWxh = dWxh.unaryExpr([max_gradient](double val) { return std::max(-max_gradient, std::min(val, max_gradient)); });
        dWhh = dWhh.unaryExpr([max_gradient](double val) { return std::max(-max_gradient, std::min(val, max_gradient)); });
        dWhy = dWhy.unaryExpr([max_gradient](double val) { return std::max(-max_gradient, std::min(val, max_gradient)); });
        dbh = dbh.unaryExpr([max_gradient](double val) { return std::max(-max_gradient, std::min(val, max_gradient)); });
        dby = dby.unaryExpr([max_gradient](double val) { return std::max(-max_gradient, std::min(val, max_gradient)); });

        // Check for NaNs after clipping
        if (dWxh.hasNaN() || dWhh.hasNaN() || dWhy.hasNaN() || dbh.hasNaN() || dby.hasNaN()) {
            std::cerr << "NaN detected in gradients after clipping" << std::endl;
            return loss;
        }
    }

    hprev = hs[inputs.size()];
    return loss;
}

// Sampling function to generate new sequences from the model
std::vector<int> sample(
    Eigen::VectorXd& h,
    int seed_ix,
    int n,
    const Eigen::MatrixXd& Wxh,
    const Eigen::MatrixXd& Whh,
    const Eigen::MatrixXd& Why,
    const Eigen::VectorXd& bh,
    const Eigen::VectorXd& by
) {
    std::vector<int> ixes;
    Eigen::VectorXd x = Eigen::VectorXd::Zero(Wxh.cols());
    x(seed_ix) = 1;

    for (int t = 0; t < n; ++t) {
        h = (Wxh * x + Whh * h + bh).array().tanh();
        Eigen::VectorXd y = Why * h + by;
        Eigen::VectorXd p = softmax(y);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> dist(p.data(), p.data() + p.size());
        int ix = dist(gen);

        x = Eigen::VectorXd::Zero(Wxh.cols());
        x(ix) = 1;
        ixes.push_back(ix);
    }

    return ixes;
}

int main() {
    // Sample text to train the RNN
    std::string data = "hello world, welcome to the world of neural networks";
    std::vector<char> chars(data.begin(), data.end());
    std::sort(chars.begin(), chars.end());
    chars.erase(std::unique(chars.begin(), chars.end()), chars.end());

    int data_size = data.size();
    int vocab_size = chars.size();

    std::unordered_map<char, int> char_to_ix;
    std::unordered_map<int, char> ix_to_char;
    for (int i = 0; i < vocab_size; ++i) {
        char_to_ix[chars[i]] = i;
        ix_to_char[i] = chars[i];
    }

    // Hyperparameters
    int hidden_size = 100;  // Size of hidden layer of neurons
    int seq_length = 25;    // Number of steps to unroll the RNN
    double learning_rate = 1e-2;  // Learning rate

    // Model parameters
    Eigen::MatrixXd Wxh(hidden_size, vocab_size);
    Eigen::MatrixXd Whh(hidden_size, hidden_size);
    Eigen::MatrixXd Why(vocab_size, hidden_size);
    Eigen::VectorXd bh = Eigen::VectorXd::Zero(hidden_size);
    Eigen::VectorXd by = Eigen::VectorXd::Zero(vocab_size);

    // Initialize weights with small values using normal distribution
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 0.001);  // Further reduce stddev to 0.001

    for (int i = 0; i < Wxh.size(); ++i) {
        Wxh(i) = distribution(generator);
    }
    for (int i = 0; i < Whh.size(); ++i) {
        Whh(i) = distribution(generator);
    }
    for (int i = 0; i < Why.size(); ++i) {
        Why(i) = distribution(generator);
    }

    // Training loop
    int n = 0;
    int p = 0;
    Eigen::MatrixXd mWxh = Eigen::MatrixXd::Zero(Wxh.rows(), Wxh.cols());
    Eigen::MatrixXd mWhh = Eigen::MatrixXd::Zero(Whh.rows(), Whh.cols());
    Eigen::MatrixXd mWhy = Eigen::MatrixXd::Zero(Why.rows(), Why.cols());
    Eigen::VectorXd mbh = Eigen::VectorXd::Zero(bh.size());
    Eigen::VectorXd mby = Eigen::VectorXd::Zero(by.size());

    double smooth_loss = -std::log(1.0 / vocab_size) * seq_length;  // Loss at iteration 0
    Eigen::VectorXd hprev = Eigen::VectorXd::Zero(hidden_size);

    while (true) {
        // Prepare inputs (we're sweeping from left to right in steps seq_length long)
        if (p + seq_length + 1 >= data_size || n == 0) {
            hprev.setZero();  // Reset RNN memory
            p = 0;  // Go from start of data
        }

        std::vector<int> inputs(seq_length);
        std::vector<int> targets(seq_length);
        for (int i = 0; i < seq_length; ++i) {
            inputs[i] = char_to_ix[data[p + i]];
            targets[i] = char_to_ix[data[p + i + 1]];
        }

        // Forward seq_length characters through the net and fetch gradient
        Eigen::MatrixXd dWxh = Eigen::MatrixXd::Zero(Wxh.rows(), Wxh.cols());
        Eigen::MatrixXd dWhh = Eigen::MatrixXd::Zero(Whh.rows(), Whh.cols());
        Eigen::MatrixXd dWhy = Eigen::MatrixXd::Zero(Why.rows(), Why.cols());
        Eigen::VectorXd dbh = Eigen::VectorXd::Zero(bh.size());
        Eigen::VectorXd dby = Eigen::VectorXd::Zero(by.size());

        double loss = lossFun(inputs, targets, Wxh, Whh, Why, bh, by, hprev, dWxh, dWhh, dWhy, dbh, dby);
        smooth_loss = smooth_loss * 0.999 + loss * 0.001;

        // Perform parameter update with Adagrad
        mWxh.array() += dWxh.array().square();
        Wxh.array() -= learning_rate * dWxh.array() / (mWxh.array().sqrt() + 1e-8);

        mWhh.array() += dWhh.array().square();
        Whh.array() -= learning_rate * dWhh.array() / (mWhh.array().sqrt() + 1e-8);

        mWhy.array() += dWhy.array().square();
        Why.array() -= learning_rate * dWhy.array() / (mWhy.array().sqrt() + 1e-8);

        mbh.array() += dbh.array().square();
        bh.array() -= learning_rate * dbh.array() / (mbh.array().sqrt() + 1e-8);

        mby.array() += dby.array().square();
        by.array() -= learning_rate * dby.array() / (mby.array().sqrt() + 1e-8);

        // Print progress and sample from the model every 1000 iterations
        if (n % 1000 == 0) {
            std::cout << "iter " << n << ", loss: " << smooth_loss << "\n";
            std::vector<int> sample_ix = sample(hprev, inputs[0], 200, Wxh, Whh, Why, bh, by);
            for (int ix : sample_ix) {
                std::cout << ix_to_char[ix];
            }
            std::cout << "\n----\n";
        }

        p += seq_length;  // Move data pointer
        n += 1;  // Iteration counter
    }

    return 0;
}
