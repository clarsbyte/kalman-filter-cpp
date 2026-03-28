#pragma once

#include <Eigen/Dense>
#include <cmath>

namespace sp {

struct SigmaPointsResult {
    Eigen::MatrixXd sigma_points;
    Eigen::VectorXd Wm;
    Eigen::VectorXd Wc;
};

class SigmaPoints {
public:
    SigmaPoints(int n, double alpha = 1e-3, double beta = 2.0, double kappa = 0.0)
        : n(n), alpha(alpha), beta(beta), kappa(kappa)
    {
        lambda = (alpha * alpha) * (n + kappa) - n;
    }

    ~SigmaPoints() {}

    SigmaPointsResult generate(const Eigen::VectorXd& x, const Eigen::MatrixXd& P) {
        SigmaPointsResult result;

        Eigen::LLT<Eigen::MatrixXd> llt(P);
        Eigen::MatrixXd L = llt.matrixL();
        Eigen::MatrixXd A = std::sqrt(n + lambda) * L;

        result.sigma_points = Eigen::MatrixXd(n, 2 * n + 1);
        result.sigma_points.col(0) = x;
        for (int i = 0; i < n; i++) {
            result.sigma_points.col(i + 1)     = x + A.col(i);
            result.sigma_points.col(i + 1 + n) = x - A.col(i);
        }

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

private:
    int n;
    double alpha, beta, kappa, lambda;

    double clampSpread(double spread) {
        if (spread < 1e-3) return 1e-3;
        else if (spread > 1) return 1;
        else return spread;
    }
};

}  // namespace sp
