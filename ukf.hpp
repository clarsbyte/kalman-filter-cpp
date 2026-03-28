#pragma once

#include <Eigen/Dense>
#include <functional>
#include "sigma_points.hpp"

namespace kf {

class UnscentedKalmanFilter {
public:
    UnscentedKalmanFilter(
        std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> f,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> h,
        const Eigen::MatrixXd& P,
        const Eigen::MatrixXd& R,
        const Eigen::MatrixXd& Q,
        double alpha = 1e-3,
        double beta  = 2.0,
        double kappa = 0.0
    )
    : f(f), h(h), P(P), P0(P), R(R), Q(Q)
    , x_hat(Eigen::VectorXd::Zero(P.rows()))
    , sigma_points(P.rows(), alpha, beta, kappa)
    {}

    ~UnscentedKalmanFilter() {}

    void init() {
        P = P0;
        x_hat.setZero();
    }

    void init(const Eigen::VectorXd& x) {
        init();
        x_hat = x;
    }

    void predict(const Eigen::VectorXd& u) {
        int n = x_hat.size();

        sp::SigmaPointsResult sp_result = sigma_points.generate(x_hat, P);
        Eigen::MatrixXd& chi = sp_result.sigma_points;
        Eigen::VectorXd& Wm  = sp_result.Wm;
        Eigen::VectorXd& Wc  = sp_result.Wc;

        Eigen::MatrixXd Y(n, 2 * n + 1);
        for (int i = 0; i < 2 * n + 1; i++)
            Y.col(i) = f(chi.col(i), u);

        x_hat = Y * Wm;

        P = Q;
        for (int i = 0; i < 2 * n + 1; i++) {
            Eigen::VectorXd diff = Y.col(i) - x_hat;
            P += Wc(i) * diff * diff.transpose();
        }
    }

    void update(const Eigen::VectorXd& z) {
        int n = x_hat.size();

        sp::SigmaPointsResult sp_result = sigma_points.generate(x_hat, P);
        Eigen::MatrixXd& Y  = sp_result.sigma_points;
        Eigen::VectorXd& Wm = sp_result.Wm;
        Eigen::VectorXd& Wc = sp_result.Wc;

        int m = h(Y.col(0)).size();
        Eigen::MatrixXd Z(m, 2 * n + 1);
        for (int i = 0; i < 2 * n + 1; i++)
            Z.col(i) = h(Y.col(i));

        Eigen::VectorXd mu_z = Z * Wm;

        Eigen::MatrixXd Pz = R;
        for (int i = 0; i < 2 * n + 1; i++) {
            Eigen::VectorXd dz = Z.col(i) - mu_z;
            Pz += Wc(i) * dz * dz.transpose();
        }

        Eigen::MatrixXd Pxz = Eigen::MatrixXd::Zero(n, m);
        for (int i = 0; i < 2 * n + 1; i++) {
            Eigen::VectorXd dx = Y.col(i) - x_hat;
            Eigen::VectorXd dz = Z.col(i) - mu_z;
            Pxz += Wc(i) * dx * dz.transpose();
        }

        K      = Pxz * Pz.inverse();
        x_hat += K * (z - mu_z);
        P     -= K * Pz * K.transpose();
    }

    Eigen::VectorXd get_state()      { return x_hat; }
    Eigen::MatrixXd get_covariance() { return P; }

private:
    std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> f;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> h;
    Eigen::MatrixXd P, R, Q, P0, K;
    Eigen::VectorXd x_hat;
    sp::SigmaPoints sigma_points;
};

}  // namespace kf
