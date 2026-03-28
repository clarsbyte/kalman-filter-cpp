#include <iostream>
#include <fstream>
#include <cmath>
#include "sigma_points.hpp"

int main(){
    int n = 2;  // 2D for better visualization
    // UKF parameters: alpha controls spread, beta=2 for Gaussian, kappa typically 3-n
    sp::SigmaPoints sp(n, 1.0, 2.0, 1.0);
    Eigen::VectorXd x(n);
    x << 0.0, 0.0;

    // Covariance matrix - create an ellipse
    Eigen::MatrixXd P(n, n);
    P << 1.0, 0.3,
         0.3, 0.5;

    sp::SigmaPointsResult result = sp.generate(x, P);

    // Write sigma points to file
    std::ofstream outfile("sigma_points.csv");
    outfile << "x,y,type\n";

    // Mean point
    outfile << x(0) << "," << x(1) << ",mean\n";

    // Sigma points
    for (int i = 0; i < result.sigma_points.cols(); i++) {
        outfile << result.sigma_points(0, i) << ","
                << result.sigma_points(1, i) << ",sigma\n";
    }

    // Write covariance matrix for ellipse calculation
    std::ofstream covfile("covariance.csv");
    covfile << P(0, 0) << "," << P(0, 1) << "\n";
    covfile << P(1, 0) << "," << P(1, 1) << "\n";

    std::cout << "Sigma Points:\n" << result.sigma_points << std::endl;
    std::cout << "Weights (mean):\n" << result.Wm.transpose() << std::endl;
    std::cout << "Weights (covariance):\n" << result.Wc.transpose() << std::endl;
    std::cout << "\nData written to sigma_points.csv and covariance.csv\n";
    std::cout << "Run: python3 plot_sigma_points.py\n";
}