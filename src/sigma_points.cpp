#include "sigma_points.hpp"
#include <cmath>

sp::SigmaPoints::SigmaPoints(int _n, double _alpha, double _beta, double _kappa)
    : n(_n), alpha(_alpha), beta(_beta), kappa(_kappa) {
    // computing lambda
    lambda = (alpha * alpha) * (n + kappa) - n;
}

sp::SigmaPoints::~SigmaPoints() {
}

double sp::SigmaPoints::clampSpread(double spread) {
    if (spread < 1e-3) return 1e-3;
    else if (spread > 1) return 1;
    else return spread;
}

sp::SigmaPointsResult sp::SigmaPoints::generate(const Eigen::VectorXd& x, const Eigen::MatrixXd& P) {
    SigmaPointsResult result;

    // Compute scaled Cholesky decomposition
    Eigen::LLT<Eigen::MatrixXd> llt(P);
    Eigen::MatrixXd L = llt.matrixL();
    Eigen::MatrixXd A = std::sqrt(n + lambda) * L;

    // Generate sigma points
    result.sigma_points = Eigen::MatrixXd(n, 2 * n + 1);
    result.sigma_points.col(0) = x;

    for (int i = 0; i < n; i++) {
        result.sigma_points.col(i + 1) = x + A.col(i);
        result.sigma_points.col(i + 1 + n) = x - A.col(i);
    }

    // weights
    result.Wm = Eigen::VectorXd(2 * n + 1);
    result.Wc = Eigen::VectorXd(2 * n + 1);

    result.Wm(0) = lambda / (n + lambda);
    result.Wc(0) = lambda / (n + lambda) + (1 - alpha * alpha + beta);

    double w = 1.0 / (2.0 * (n + lambda));
    for (int i = 1; i < 2 * n + 1; i++) {
        result.Wm(i) = w;
        result.Wc(i) = w;
    }

    return result;
}

