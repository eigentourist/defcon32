#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::MatrixXd matA(2, 2);
    matA(0, 0) = 1;
    matA(1, 0) = 2;
    matA(0, 1) = 3;
    matA(1, 1) = 4;

    std::cout << "Here is the matrix A:\n" << matA << std::endl;

    Eigen::MatrixXd matB = matA.transpose();
    std::cout << "Here is the transpose of A:\n" << matB << std::endl;

    return 0;
}
