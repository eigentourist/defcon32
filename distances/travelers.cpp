#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>

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

// Function to calculate the covariance matrix
std::vector<std::vector<double>> calculateCovarianceMatrix(const std::vector<std::vector<double>> &data) {
    size_t n = data.size();
    size_t m = data[0].size();

    std::vector<std::vector<double>> covarianceMatrix(m, std::vector<double>(m, 0.0));
    std::vector<double> means(m, 0.0);

    // Calculate means
    for (const auto &row : data) {
        for (size_t j = 0; j < m; ++j) {
            means[j] += row[j];
        }
    }
    for (size_t j = 0; j < m; ++j) {
        means[j] /= n;
    }

    // Calculate covariance matrix
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = i; j < m; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += (data[k][i] - means[i]) * (data[k][j] - means[j]);
            }
            covarianceMatrix[i][j] = sum / (n - 1);
            covarianceMatrix[j][i] = covarianceMatrix[i][j]; // Symmetric matrix
        }
    }

    return covarianceMatrix;
}

// Function to calculate eigenvalues and eigenvectors (Placeholder for simplicity)
std::pair<std::vector<double>, std::vector<std::vector<double>>> calculateEigen(const std::vector<std::vector<double>> &covarianceMatrix) {
    // Placeholder for eigenvalue and eigenvector calculations.
    // You can use specialized libraries like Eigen or Armadillo for this purpose.

    // Eigen library: 
    // - URL: https://eigen.tuxfamily.org
    // - Installation: You can download the latest version from the website or install it via package managers like apt (for Ubuntu) using `sudo apt-get install libeigen3-dev`.
    // - Eigen is open-source and licensed under the Mozilla Public License 2.0 (MPL-2.0).

    // Armadillo library:
    // - URL: http://arma.sourceforge.net
    // - Installation: Available via package managers (e.g., `sudo apt-get install libarmadillo-dev` on Ubuntu) or you can download the source code from the website.
    // - Armadillo is open-source and can be used under the Apache License 2.0 or GNU GPL 2+.


    size_t m = covarianceMatrix.size();
    std::vector<double> eigenvalues(m, 1.0); // Placeholder values
    std::vector<std::vector<double>> eigenvectors(m, std::vector<double>(m, 0.0));

    for (size_t i = 0; i < m; ++i) {
        eigenvectors[i][i] = 1.0; // Identity matrix as placeholder
    }

    return {eigenvalues, eigenvectors};
}

// Function to perform PCA
std::vector<std::vector<double>> performPCA(const std::vector<std::vector<double>> &data, int reducedDimensions) {
    std::vector<std::vector<double>> covarianceMatrix = calculateCovarianceMatrix(data);
    auto [eigenvalues, eigenvectors] = calculateEigen(covarianceMatrix);

    // Sort eigenvectors by eigenvalues
    std::vector<size_t> indices(eigenvalues.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&eigenvalues](size_t i1, size_t i2) {
        return eigenvalues[i1] > eigenvalues[i2];
    });

    // Select top 'reducedDimensions' eigenvectors
    std::vector<std::vector<double>> selectedEigenvectors(reducedDimensions);
    for (int i = 0; i < reducedDimensions; ++i) {
        selectedEigenvectors[i] = eigenvectors[indices[i]];
    }

    // Project data onto the new reduced dimensions
    std::vector<std::vector<double>> reducedData(data.size(), std::vector<double>(reducedDimensions, 0.0));
    for (size_t i = 0; i < data.size(); ++i) {
        for (int j = 0; j < reducedDimensions; ++j) {
            for (size_t k = 0; k < data[0].size(); ++k) {
                reducedData[i][j] += data[i][k] * selectedEigenvectors[j][k];
            }
        }
    }

    return reducedData;
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
