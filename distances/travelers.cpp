#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense> 

// Function to load CSV file into a matrix
std::vector<std::vector<double>> loadCSV(const std::string &filename) {
    std::ifstream file(filename);
    std::vector<std::vector<double>> data;

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }

    std::string line;
    bool firstLine = true; // Assuming the first line is a header

    while (std::getline(file, line)) {
        // Skip the header row
        if (firstLine) {
            firstLine = false;
            continue;
        }

        std::vector<double> row;
        std::stringstream ss(line);
        std::string value;

        while (std::getline(ss, value, ',')) {
            try {
                // Attempt to convert each value to a double
                row.push_back(std::stod(value));
            } catch (const std::invalid_argument& e) {
                std::cerr << "Error: Conversion error for value '" << value << "' in line: " << line << std::endl;
                return {};  // Return an empty vector on error
            } catch (const std::out_of_range& e) {
                std::cerr << "Error: Value out of range for conversion '" << value << "' in line: " << line << std::endl;
                return {};  // Return an empty vector on error
            }
        }

        if (!row.empty()) {
            data.push_back(row);
        }
    }

    file.close();
    return data;
}

// Function to save matrix to CSV file
void saveCSV(const std::string &filename, const std::vector<std::vector<double>> &data, const std::vector<std::string> &header) {
    std::ofstream file(filename);

    // Write header
    for (size_t i = 0; i < header.size(); ++i) {
        file << header[i];
        if (i < header.size() - 1) {
            file << ",";
        }
    }
    file << "\n";

    // Write data
    for (const auto &row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
}

// Function to calculate the covariance matrix using Eigen
Eigen::MatrixXd calculateCovarianceMatrix(const Eigen::MatrixXd &data) {
    Eigen::MatrixXd centered = data.rowwise() - data.colwise().mean();
    Eigen::MatrixXd cov = (centered.adjoint() * centered) / double(data.rows() - 1);
    return cov;
}

// Function to perform eigenvalue and eigenvector calculations using Eigen
std::pair<Eigen::VectorXd, Eigen::MatrixXd> calculateEigen(const Eigen::MatrixXd &covarianceMatrix) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(covarianceMatrix);
    return {solver.eigenvalues().reverse(), solver.eigenvectors().rowwise().reverse()};
}

// Function to perform PCA
std::vector<std::vector<double>> performPCA(const std::vector<std::vector<double>> &data, int reducedDimensions) {
    Eigen::MatrixXd dataMatrix = Eigen::MatrixXd::Zero(data.size(), data[0].size());

    // Convert std::vector<std::vector<double>> to Eigen::MatrixXd
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[0].size(); ++j) {
            dataMatrix(i, j) = data[i][j];
        }
    }

    Eigen::MatrixXd covarianceMatrix = calculateCovarianceMatrix(dataMatrix);
    auto [eigenvalues, eigenvectors] = calculateEigen(covarianceMatrix);

    // Select the top 'reducedDimensions' eigenvectors
    Eigen::MatrixXd selectedEigenvectors = eigenvectors.leftCols(reducedDimensions);

    // Project the data onto the new reduced dimensions
    Eigen::MatrixXd reducedData = dataMatrix * selectedEigenvectors;

    // Convert Eigen::MatrixXd back to std::vector<std::vector<double>>
    std::vector<std::vector<double>> reducedDataVector(reducedData.rows(), std::vector<double>(reducedDimensions));
    for (size_t i = 0; i < reducedData.rows(); ++i) {
        for (int j = 0; j < reducedDimensions; ++j) {
            reducedDataVector[i][j] = reducedData(i, j);
        }
    }

    return reducedDataVector;
}

int main() {
    // Load data
    std::vector<std::vector<double>> data = loadCSV("data/travelers_train.csv");

    // Perform PCA to reduce to 3 dimensions
    int reducedDimensions = 3;
    std::vector<std::vector<double>> reducedData = performPCA(data, reducedDimensions);

    // Save the reduced dataset
    std::vector<std::string> header = {"epoch_wave", "dimension_fold", "continuum_shift"};
    saveCSV("data/reduced_travelers.csv", reducedData, header);

    std::cout << "PCA completed. Reduced dataset saved as 'data/reduced_travelers.csv'." << std::endl;

    return 0;
}
